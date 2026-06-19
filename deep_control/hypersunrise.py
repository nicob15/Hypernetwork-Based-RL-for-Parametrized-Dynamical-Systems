import argparse
import copy
import math
import os
from itertools import chain
import random
import time
import wandb
import argparse
import numpy as np
import tensorboardX
import torch
import torch.nn.functional as F
import tqdm

from . import nets, replay, run, utils
from .utils import device


class SunriseAgent:
    def __init__(

        self,
        obs_space_size,
        act_space_size,
        log_std_low,
        log_std_high,
        ensemble_size=5,
        ucb_bonus=5.0,
        actor_net_cls=None,
        critic_net_cls=None,
    ):
        if actor_net_cls is None:
            actor_net_cls = nets.HyperStochasticActor
        if critic_net_cls is None:
            critic_net_cls = nets.HyperCritic
        # Usa lo stesso device definito in utils
        self.device = device

        self.obs_size = obs_space_size
        self.act_size = act_space_size

        self.actors = [
            actor_net_cls(
                obs_space_size,
                act_space_size,
                log_std_low,
                log_std_high,
                dist_impl="pyd",
            ).to(self.device)
            for _ in range(ensemble_size)
        ]

        self.critics = [
            critic_net_cls(obs_space_size, act_space_size).to(self.device)
            for _ in range(ensemble_size)
        ]

        # SUNRISE Eq.6 lambda (UCB coefficient)
        self.ucb_bonus = ucb_bonus

    # --------------------------------------------------
    def to(self, dev):
        self.device = dev
        for i, (actor, critic) in enumerate(zip(self.actors, self.critics)):
            self.critics[i] = critic.to(dev)
            self.actors[i] = actor.to(dev)

    # --------------------------------------------------
    def eval(self):
        for actor, critic in zip(self.actors, self.critics):
            critic.eval()
            actor.eval()

    def train(self):
        for actor, critic in zip(self.actors, self.critics):
            critic.train()
            actor.train()

    # --------------------------------------------------
    def save(self, path):
        for i, (actor, critic) in enumerate(zip(self.actors, self.critics)):
            actor_path = os.path.join(path, f"actor{i}.pt")
            critic_path = os.path.join(path, f"critic{i}.pt")
            torch.save(actor.state_dict(), actor_path)
            torch.save(critic.state_dict(), critic_path)

    def load(self, path):
        for i, (actor, critic) in enumerate(zip(self.actors, self.critics)):
            actor_path = os.path.join(path, f"actor{i}.pt")
            critic_path = os.path.join(path, f"critic{i}.pt")
            actor.load_state_dict(torch.load(actor_path, weights_only=False, map_location=self.device))
            critic.load_state_dict(torch.load(critic_path, weights_only=False, map_location=self.device))

    # --------------------------------------------------
    def forward(self, state, from_cpu=True):
        # evaluation forward: media delle mean dei vari actor
        if from_cpu:
            state = self.process_state(state)
        self.eval()
        with torch.no_grad():
            act = torch.stack(
                [actor.forward(state).mean for actor in self.actors], dim=0
            ).mean(0)
        self.train()
        if from_cpu:
            act = self.process_act(act)
        return act

    # --------------------------------------------------
    def sample_action(self, state, from_cpu=True):
        """
        UCB exploration come descritto nel paper SUNRISE.
        Ritorna una singola action numpy 1D di shape (act_dim,).
        """
        # state: numpy array oppure torch
        if from_cpu:
            state_t = self.process_state(state)  # (1, obs_dim) on device
        else:
            if isinstance(state, np.ndarray):
                state_t = torch.from_numpy(state).float()
            elif isinstance(state, torch.Tensor):
                state_t = state.float()
            else:
                raise TypeError("State is neither numpy array nor torch tensor")
            if state_t.dim() == 1:
                state_t = state_t.unsqueeze(0)
            state_t = state_t.to(self.device)

        self.eval()
        with torch.no_grad():
            # candidate actions da ogni actor: shape (ensemble, act_dim)
            act_candidates = torch.stack(
                [actor.forward(state_t).sample().squeeze(0) for actor in self.actors],
                dim=0,
            )
            # valutiamo ogni azione con ogni critic: shape (n_critics, n_actions, 1)
            q_vals = torch.stack(
                [
                    critic(state_t.repeat(len(act_candidates), 1), act_candidates)
                    for critic in self.critics
                ],
                dim=0,
            )
            # SUNRISE: UCB = mean + lambda * std (over critics)
            ucb_val = q_vals.mean(0) + self.ucb_bonus * q_vals.std(0)
            # scegliamo l'azione con UCB massimo
            argmax_ucb_val = torch.argmax(ucb_val)
            act = act_candidates[argmax_ucb_val].unsqueeze(0)  # shape (1, act_dim)
        self.train()

        # clamp in [-1, 1] e converte in numpy 1D
        act_np = self.process_act(act)
        return act_np

    # --------------------------------------------------
    def forward_lcb(self, state, from_cpu=True):
        """
        LCB evaluation: come sample_action ma sceglie l'azione che
        MINIMIZZA l'incertezza dell'ensemble di critic.
        Seleziona: argmax( mean_Q - ucb_bonus * std_Q )
        ovvero la candidata con il Q più alto fra quelle a bassa incertezza.
        """
        if from_cpu:
            state_t = self.process_state(state)
        else:
            if isinstance(state, np.ndarray):
                state_t = torch.from_numpy(state).float()
            elif isinstance(state, torch.Tensor):
                state_t = state.float()
            else:
                raise TypeError("State is neither numpy array nor torch tensor")
            if state_t.dim() == 1:
                state_t = state_t.unsqueeze(0)
            state_t = state_t.to(self.device)

        self.eval()
        with torch.no_grad():
            # mean di ogni actor (deterministico, come forward)
            act_candidates = torch.stack(
                [actor.forward(state_t).mean.squeeze(0) for actor in self.actors],
                dim=0,
            )  # (ensemble, act_dim)
            # Q di ogni critic per ogni candidata: (n_critics, n_actions, 1)
            q_vals = torch.stack(
                [
                    critic(state_t.repeat(len(act_candidates), 1), act_candidates)
                    for critic in self.critics
                ],
                dim=0,
            )
            # LCB = mean_Q - lambda * std_Q  → favorisce bassa incertezza
            lcb_val = q_vals.mean(0) - self.ucb_bonus * q_vals.std(0)
            act = act_candidates[torch.argmax(lcb_val)].unsqueeze(0)
        self.train()

        if from_cpu:
            act = self.process_act(act)
        return act

    # --------------------------------------------------
    def process_state(self, state):
     state = np.asarray(state, dtype=np.float32)

     # Se è già (obs_dim,) → ok
     if state.ndim == 1:
        state = np.expand_dims(state, 0)

     # Se è (1, obs_dim) → ok
     if state.ndim == 2 and state.shape[0] == 1:
        return torch.from_numpy(state).float().to(self.device)

     # Se è (1,1,obs_dim) → correggi
     if state.ndim == 3 and state.shape[0] == 1 and state.shape[1] == 1:
        state = state.reshape(1, -1)
        return torch.from_numpy(state).float().to(self.device)

     raise ValueError(f"Unexpected state shape in process_state: {state.shape}")


    def process_act(self, act):
        # act: tensor (1, act_dim) → numpy (act_dim,)
        return np.squeeze(act.clamp(-1.0, 1.0).cpu().numpy(), 0)
    




def learn_sunrise(
    buffer,
    target_agent,
    agent,
    critic_optimizer,
    batch_size,
    gamma,
    critic_clip,
    actor_optimizer,
    alpha_optimizers,
    target_entropy,
    log_alphas,
    actor_clip,
    weighted_bellman_temp,
):
    per = isinstance(buffer, replay.PrioritizedReplayBuffer)
    if per:
        batch, imp_weights, priority_idxs = buffer.sample(batch_size)
        imp_weights = imp_weights.to(device)
    else:
        batch = buffer.sample(batch_size)

    state_batch, action_batch, reward_batch, next_state_batch, done_batch = batch
    state_batch = state_batch.to(device)
    next_state_batch = next_state_batch.to(device)
    action_batch = action_batch.to(device)
    reward_batch = reward_batch.to(device)
    done_batch = done_batch.to(device)

    agent.train()

    ###################
    ## CRITIC UPDATE ##
    ###################

    with torch.no_grad():
        # compute weighted bellman coeffs using SUNRISE Eq 5
        target_q_std = torch.stack(
            [q(state_batch, action_batch) for q in target_agent.critics], dim=0
        ).std(0)
        weights = torch.sigmoid(-target_q_std * weighted_bellman_temp) + 0.5

    # now we compute the MSBE of each critic relative to its own target
    critic_loss = 0.0
    total_abs_td_error = 0.0
    for i, critic in enumerate(agent.critics):
        with torch.no_grad():
            # sample an action from actor i
            action_dist_s1 = agent.actors[i](next_state_batch)
            action_s1 = action_dist_s1.rsample()
            logp_a1 = action_dist_s1.log_prob(action_s1).sum(-1, keepdim=True)
            # generate target network's Q(s', a') prediction
            target_q_s1 = target_agent.critics[i](next_state_batch, action_s1)
            # compute TD target value for this critic
            td_target = reward_batch + gamma * (1.0 - done_batch) * (
                target_q_s1 - (log_alphas[i].exp() * logp_a1)
            )
        # compute MSBE for this critic
        agent_critic_pred = critic(state_batch, action_batch)

        huber = F.smooth_l1_loss(
        agent_critic_pred,
        td_target,
        reduction="none"
        )

        critic_loss += weights * huber
        total_abs_td_error += torch.abs(td_target - agent_critic_pred)

    if per:
        # priority weights can be used in addition to bellman backup weights.
        critic_loss *= imp_weights
    critic_loss = critic_loss.mean()
    critic_optimizer.zero_grad()
    critic_loss.backward()
    if critic_clip:
        torch.nn.utils.clip_grad_norm_(
            chain(*(critic.parameters() for critic in agent.critics)), critic_clip
        )
    critic_optimizer.step()

    ##########################
    ## ACTOR + ALPHA UPDATE ##
    ##########################
    actor_loss = 0.0
    for i, actor in enumerate(agent.actors):
        # sample an action for this actor
        dist = actor(state_batch)
        agent_actions = dist.rsample()
        logp_a = dist.log_prob(agent_actions).sum(-1, keepdim=True)
        # use corresponding critic to evaluate this action
        critic_pred = agent.critics[i](state_batch, agent_actions)
        actor_loss += -(critic_pred - (log_alphas[i].exp().detach() * logp_a)).mean()

        # update alpha_i
        alpha_loss = (-log_alphas[i].exp() * (logp_a + target_entropy).detach()).mean()
        alpha_optimizers[i].zero_grad()
        alpha_loss.backward()
        alpha_optimizers[i].step()

    # actor gradient step
    actor_optimizer.zero_grad()
    actor_loss.backward()
    if actor_clip:
        torch.nn.utils.clip_grad_norm_(
            chain(*(actor.parameters() for actor in agent.actors)), actor_clip
        )
    actor_optimizer.step()

    if per:
        ensemble_size = float(len(agent.actors))
        avg_abs_td_error = total_abs_td_error / ensemble_size
        new_priorities = (avg_abs_td_error + 1e-5).cpu().detach().squeeze(1).numpy()
        buffer.update_priorities(priority_idxs, new_priorities)
    return {
    "critic_loss": critic_loss.item(),
    "actor_loss": actor_loss.item(),
    }