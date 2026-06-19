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
    def __init__(self, obs_space_size, act_space_size, ensemble_size=5, ucb_bonus=5.0, actor_net_cls=None,
                 critic_net_cls=None,):
        if actor_net_cls is None:
            actor_net_cls = nets.HyperActor
        if critic_net_cls is None:
            critic_net_cls = nets.HyperCritic
        # Usa lo stesso device definito in utils
        self.device = device

        self.obs_size = obs_space_size
        self.act_size = act_space_size

        self.actors = [
            actor_net_cls(obs_space_size, act_space_size,).to(self.device)
            for _ in range(ensemble_size)
        ]

        self.actors_target = [
            actor_net_cls(obs_space_size, act_space_size,).to(self.device)
            for _ in range(ensemble_size)
        ]

        self.critics = [
            critic_net_cls(obs_space_size, act_space_size).to(self.device)
            for _ in range(ensemble_size)
        ]

        self.critics_target = [
            critic_net_cls(obs_space_size, act_space_size).to(self.device)
            for _ in range(ensemble_size)
        ]

        for i in range(ensemble_size):
            self.copy_params(self.actors_target[i], self.actors[i])
            self.copy_params(self.critics_target[i], self.critics[i])

        # SUNRISE Eq.6 lambda (UCB coefficient)
        self.ucb_bonus = ucb_bonus

    def copy_params(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

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
            actor.load_state_dict(torch.load(actor_path))
            critic.load_state_dict(torch.load(critic_path))

    # --------------------------------------------------
    def forward(self, state, from_cpu=True):
        # evaluation forward: media delle mean dei vari actor
        if from_cpu:
            state = self.process_state(state)
        self.eval()
        with torch.no_grad():
            act = torch.stack(
                [actor.forward(state) for actor in self.actors], dim=0
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
                [(actor.forward(state_t) + torch.normal(mean=0, std=0.1, size=(1, actor.act_space_size)).to(self.device)).clip(-1.0, 1.0).squeeze(0) for actor in self.actors],
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


def learn_sunrise(buffer, target_agent, agent, critic_optimizer, batch_size, gamma, critic_clip, actor_optimizer,
                  alpha_optimizers, target_entropy, log_alphas, actor_clip, weighted_bellman_temp, it, policy_freq=2, tau=0.005):
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
        target_q_std = torch.stack([q(state_batch, action_batch) for q in target_agent.critics], dim=0).std(0)
        weights = torch.sigmoid(-target_q_std * weighted_bellman_temp) + 0.5

    # now we compute the MSBE of each critic relative to its own target
    critic_loss = 0.0
    total_abs_td_error = 0.0
    for i, critic in enumerate(agent.critics):
        with torch.no_grad():

            next_action_batch = agent.actors_target[i](next_state_batch.detach())

            # Compute the target Q value
            target_Q = agent.critics_target[i](next_state_batch, next_action_batch)
            td_target = reward_batch + gamma * (1.0 - done_batch) * target_Q

        # compute MSBE for this critic
        agent_critic_pred = critic(state_batch, action_batch)

        huber = F.smooth_l1_loss(agent_critic_pred, td_target, reduction="none")

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
    # Delayed policy updates
    actor_loss = torch.zeros_like(critic_loss)
    if it % policy_freq == 0:
        actor_loss = 0.0
        for i, actor in enumerate(agent.actors):

            actor_loss += -agent.critics[i](state_batch.detach(), actor(state_batch)).mean()

        # actor gradient step
        actor_optimizer.zero_grad()
        disable_gradients(agent.critics)
        actor_loss.backward()
        if actor_clip:
            torch.nn.utils.clip_grad_norm_(
                chain(*(actor.parameters() for actor in agent.actors)), actor_clip
            )
        enable_gradients(agent.critics)
        actor_optimizer.step()


        # Update the frozen target models
        for i, actor in enumerate(agent.actors):
            for param, target_param in zip(agent.actors[i].parameters(), agent.actors_target[i].parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

            for param, target_param in zip(agent.critics[i].parameters(), agent.critics_target[i].parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    if per:
        ensemble_size = float(len(agent.actors))
        avg_abs_td_error = total_abs_td_error / ensemble_size
        new_priorities = (avg_abs_td_error + 1e-5).cpu().detach().squeeze(1).numpy()
        buffer.update_priorities(priority_idxs, new_priorities)
    return {
        "critic_loss": critic_loss.item(),
        "actor_loss": actor_loss.item(),
    }


def disable_gradients(critics):
    for critic in critics:
        for p in critic.parameters():
            p.requires_grad = False


def enable_gradients(critics):
    for critic in critics:
        for p in critic.parameters():
            p.requires_grad = True