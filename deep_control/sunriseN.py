import argparse
import copy
import math
import os
from itertools import chain
import random
import time
import wandb

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
        actor_net_cls=nets.StochasticActor,
        critic_net_cls=nets.BigCritic,
    ):
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
    

def evaluate_sunrise(agent, env, episodes, max_steps):
    avg_reward = 0.
    avg_state_cost = 0.
    avg_action_cost = 0.

    for ep in range(episodes):
        s, done = env.reset(), False
        ep_state_cost = 0.
        ep_action_cost = 0.
        ep_reward = 0.
        steps = 0

        while not done:
            a = agent.sample_action(s)
            s, r, done, info = env.step(a)

            ep_reward += r
            ep_state_cost += info["state_cost"]
            ep_action_cost += info["action_cost"]
            steps += 1

        avg_reward += ep_reward
        avg_state_cost += ep_state_cost
        avg_action_cost += ep_action_cost

    avg_reward /= episodes
    avg_state_cost /= episodes
    avg_action_cost /= episodes

    print("---------------------------------------")
    print(f"Evaluation over {episodes} episodes:")
    print(f"  Avg Reward      : {avg_reward:.4f}")
    print(f"  Avg State Cost  : {avg_state_cost:.4f}")
    print(f"  Avg Action Cost : {avg_action_cost:.4f}")
    print("---------------------------------------")

    return avg_reward, avg_state_cost, avg_action_cost


def sunrise(
    agent,
    buffer,
    train_env,
    test_env,
    num_steps=1_000_000,
    transitions_per_step=1,
    max_episode_steps=100_000,
    batch_size=512,
    tau=0.005,
    actor_lr=1e-4,
    critic_lr=1e-4,
    alpha_lr=1e-4,
    gamma=0.99,
    eval_interval=5000,
    eval_episodes=10,
    warmup_steps=1000,
    actor_clip=None,
    critic_clip=None,
    actor_l2=0.0,
    critic_l2=0.0,
    target_delay=2,
    save_interval=100_000,
    name="sunrise_run",
    render=False,
    save_to_disk=True,
    log_to_disk=True,
    verbosity=0,
    gradient_updates_per_step=1,
    init_alpha=0.1,
    weighted_bellman_temp=20.0,
    infinite_bootstrap=True,
    **kwargs,
):
    """
    "SUNRISE: A Simple Unified Framework for Ensemble Learning 
    in Deep Reinforcement Learning", Lee et al., 2020.
    """

    save_dir = os.path.join("save_models", "sunrise", name)

    # -----------------------------
    # INIT WANDB (come train_ks.py)
    # -----------------------------
    wandb_logging = bool(kwargs.get("log", True))

    if wandb_logging:
        config = {
            "algorithm": "sunrise",
            "ensemble_size": len(agent.actors),
            "ucb_bonus": agent.ucb_bonus,
            "num_steps": num_steps,
            "max_episode_steps": max_episode_steps,
            "batch_size": batch_size,
            "actor_lr": actor_lr,
            "critic_lr": critic_lr,
            "gamma": gamma,
            "warmup_steps": warmup_steps,
            "eval_interval": eval_interval,
            "eval_episodes": eval_episodes,
            "action_dim": train_env.action_space.shape[0],
            "obs_dim": train_env.observation_space.shape[0],
        }

        wandb.init(
            project="Control of PDE with DRL",
            tags=["KuramotoSivashinsky", "SUNRISE"],
            config=config,
            settings={"_service_wait": 600, "init_timeout": 600},
        )

    if save_to_disk:
        os.makedirs(save_dir, exist_ok=True)
    if log_to_disk:
        writer = tensorboardX.SummaryWriter(save_dir)
        writer.add_hparams(locals(), {})

    ###########
    ## SETUP ##
    ###########
    agent.to(device)
    agent.train()
    target_agent = copy.deepcopy(agent)

    # initialize all of the critic targets
    for target_critic, agent_critic in zip(target_agent.critics, agent.critics):
        utils.hard_update(target_critic, agent_critic)
    target_agent.train()

    critic_optimizer = torch.optim.Adam(
        chain(*(critic.parameters() for critic in agent.critics)),
        lr=critic_lr,
        weight_decay=critic_l2,
        betas=(0.9, 0.999),
    )
    actor_optimizer = torch.optim.Adam(
        chain(*(actor.parameters() for actor in agent.actors)),
        lr=actor_lr,
        weight_decay=actor_l2,
        betas=(0.9, 0.999),
    )

    # create a separate entropy coeff for each agent in the ensemble
    log_alphas, alpha_optimizers = [], []
    for _ in range(len(agent.actors)):
        log_alpha = torch.Tensor([math.log(init_alpha)]).to(device)
        log_alpha.requires_grad = True
        alpha_optimizer = torch.optim.Adam(
            [log_alpha], lr=alpha_lr, betas=(0.5, 0.999)
        )
        log_alphas.append(log_alpha)
        alpha_optimizers.append(alpha_optimizer)
    target_entropy = -train_env.action_space.shape[0]

    ###################
    ## TRAINING LOOP ##
    ###################
    run.warmup_buffer(buffer, train_env, warmup_steps, max_episode_steps)

    done = True
    episode = 0
    episode_reward = 0.0
    episode_state_cost = 0.0
    episode_action_cost = 0.0
    steps_this_ep = 0
    episode_start_time = time.time()

    for step in range(num_steps):

        # ===== SE INIZIA UN NUOVO EPISODIO =====
        if done:
            if episode > 0:
                ep_time = time.time() - episode_start_time
                print(
                    f"Episode Num: {episode} "
                    f"Step: {steps_this_ep} "
                    f"Reward: {episode_reward:.3f} "
                    f"Time: {ep_time:.2f}"
                )

                # Log episodico su wandb (come train_ks)
                if wandb_logging:
                    wandb.log(
                        {
                            "train/training_reward": episode_reward,
                            "train/training_state_cost": episode_state_cost,
                            "train/training_action_cost": episode_action_cost,
                            "train/episode": episode,
                        },
                        step=episode,
                    )

                # log anche su tensorboard, se attivo
                if log_to_disk:
                    writer.add_scalar("train/reward", episode_reward, episode)
                    writer.add_scalar(
                        "train/state_cost", episode_state_cost, episode
                    )
                    writer.add_scalar(
                        "train/action_cost", episode_action_cost, episode
                    )

            # reset episodio
            state = train_env.reset()
            
            state_noise = state.copy()
            n_features = min(8, state.shape[-1])
            noise = np.random.normal(0, 0.2, size=n_features)
            state_noise[0, :n_features] = state[0, :n_features] + noise

            state = state_noise
            episode_reward = 0.0
            episode_state_cost = 0.0
            episode_action_cost = 0.0
            steps_this_ep = 0
            episode += 1
            done = False
            episode_start_time = time.time()

        # ===== AGENT ACTION =====
        action = agent.sample_action(state)
        action = np.asarray(action, dtype=np.float32)

        next_state, reward, done, info = train_env.step(action)

        if infinite_bootstrap and steps_this_ep + 1 == max_episode_steps:
            done = False

        buffer.push(state, action, reward, next_state, done)

        # update stats episodio
        state = next_state
        episode_reward += reward
        episode_state_cost += info["state_cost"]
        episode_action_cost += info["action_cost"]
        steps_this_ep += 1

        if steps_this_ep >= max_episode_steps:
            done = True

        # ===== GRADIENT UPDATES =====
        for _ in range(gradient_updates_per_step):
            learn_sunrise(
                buffer=buffer,
                target_agent=target_agent,
                agent=agent,
                critic_optimizer=critic_optimizer,
                batch_size=batch_size,
                gamma=gamma,
                critic_clip=critic_clip,
                actor_optimizer=actor_optimizer,
                alpha_optimizers=alpha_optimizers,
                target_entropy=target_entropy,
                log_alphas=log_alphas,
                actor_clip=actor_clip,
                weighted_bellman_temp=weighted_bellman_temp,
            )

        # ===== TARGET NETWORK UPDATE =====
        if step % target_delay == 0:
            for (target_critic, agent_critic) in zip(
                target_agent.critics, agent.critics
            ):
                utils.soft_update(target_critic, agent_critic, tau)

        # ===== EVALUATION EVERY eval_interval EPISODES =====
        if done and (episode % eval_interval == 0):
            print("\n=== EVALUATION ===")
            mean_return = run.evaluate_agent(
                agent, test_env, eval_episodes, max_episode_steps, render
            )
            print(
                f"Eval after episode {episode}: avg return = {mean_return:.3f}\n"
            )

            if wandb_logging:
                wandb.log(
                    {
                        "eval/eval_reward": mean_return,
                        "eval/episode": episode,
                    },
                    step=episode,
                )

            if save_to_disk:
                agent.save(save_dir)
            if log_to_disk:
                writer.add_scalar("eval/return", mean_return, episode)

    return agent



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
        td_error = td_target - agent_critic_pred
        # SUNRISE Eq 4
        critic_loss += 0.5 * weights * (td_error ** 2)
        total_abs_td_error += abs(td_error)
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


def add_args(parser):
    parser.add_argument(
        "--num_steps", type=int, default=10 ** 6, help="Number of steps in training"
    )
    parser.add_argument(
        "--transitions_per_step",
        type=int,
        default=1,
        help="env transitions per training step. Defaults to 1, but will need to \
        be set higher for repaly ratios < 1",
    )
    parser.add_argument(
        "--max_episode_steps",
        type=int,
        default=100000,
        help="maximum steps per episode",
    )
    parser.add_argument(
        "--batch_size", type=int, default=512, help="training batch size"
    )
    parser.add_argument(
        "--tau", type=float, default=0.005, help="for model parameter % update"
    )
    parser.add_argument(
        "--actor_lr", type=float, default=3e-4, help="actor learning rate"
    )
    parser.add_argument(
        "--critic_lr", type=float, default=3e-4, help="critic learning rate"
    )
    parser.add_argument(
        "--gamma", type=float, default=0.99, help="gamma, the discount factor"
    )
    parser.add_argument(
        "--init_alpha",
        type=float,
        default=0.1,
        help="initial entropy regularization coefficeint.",
    )
    parser.add_argument(
        "--alpha_lr",
        type=float,
        default=1e-4,
        help="alpha (entropy regularization coefficeint) learning rate",
    )
    parser.add_argument(
        "--buffer_size", type=int, default=1_000_000, help="replay buffer size"
    )
    parser.add_argument(
        "--eval_interval",
        type=int,
        default=5000,
        help="how often to test the agent without exploration (in episodes)",
    )
    parser.add_argument(
        "--eval_episodes",
        type=int,
        default=10,
        help="how many episodes to run for when testing",
    )
    parser.add_argument(
        "--warmup_steps", type=int, default=1000, help="warmup length, in steps"
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="flag to enable env rendering during training",
    )
    parser.add_argument(
        "--actor_clip",
        type=float,
        default=None,
        help="gradient clipping for actor updates",
    )
    parser.add_argument(
        "--critic_clip",
        type=float,
        default=None,
        help="gradient clipping for critic updates",
    )
    parser.add_argument(
        "--name", type=str, default="redq_run", help="dir name for saves"
    )
    parser.add_argument(
        "--actor_l2",
        type=float,
        default=0.0,
        help="L2 regularization coeff for actor network",
    )
    parser.add_argument(
        "--critic_l2",
        type=float,
        default=0.0,
        help="L2 regularization coeff for critic network",
    )
    parser.add_argument(
        "--target_delay",
        type=int,
        default=2,
        help="How many training steps to go between target network updates",
    )
    parser.add_argument(
        "--save_interval",
        type=int,
        default=100_000,
        help="How many steps to go between saving the agent params to disk",
    )
    parser.add_argument(
        "--verbosity",
        type=int,
        default=1,
        help="verbosity > 0 displays a progress bar during training",
    )
    parser.add_argument(
        "--gradient_updates_per_step",
        type=int,
        default=1,
        help="how many gradient updates to make per training step",
    )
    parser.add_argument(
        "--prioritized_replay",
        action="store_true",
        help="flag that enables use of prioritized experience replay",
    )
    parser.add_argument(
        "--skip_save_to_disk",
        action="store_true",
        help="flag to skip saving agent params to disk during training",
    )
    parser.add_argument(
        "--skip_log_to_disk",
        action="store_true",
        help="flag to skip saving agent performance logs to disk durante il training",
    )
    parser.add_argument(
        "--log_std_low",
        type=float,
        default=-10,
        help="Lower bound for log std of action distribution.",
    )
    parser.add_argument(
        "--log_std_high",
        type=float,
        default=2,
        help="Upper bound for log std of action distribution.",
    )
    parser.add_argument(
        "--ensemble_size", type=int, default=5, help="SUNRISE ensemble size",
    )
    parser.add_argument(
        "--ucb_bonus",
        type=float,
        default=5.0,
        help="coeff for std term in ucb exploration. higher values prioritize exploring uncertain actions",
    )
    parser.add_argument(
        "--weighted_bellman_temp",
        type=float,
        default=20.0,
        help="temperature in sunrise's weight adjustment. See equation 5 of the sunrise paper",
    )
    
