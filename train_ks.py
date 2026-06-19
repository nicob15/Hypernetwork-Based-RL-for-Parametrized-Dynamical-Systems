import numpy as np
import os
import copy
import math
import time
from itertools import chain

import torch
import wandb
import argparse

from envs.ks import KuramotoSivashinskyEnv
from agents.td3 import TD3
from agents.hypeRL_td3 import TD3 as hypeRLTD3
from agents.polyL0_td3 import TD3 as polyL0TD3
from utils.utils import ReplayBuffer as TrainingBuffer
from deep_control.sunrise import SunriseAgent, learn_sunrise
from deep_control.hypersunrise import SunriseAgent as HyperSunriseAgent, learn_sunrise as learn_hypersunrise
from deep_control import replay
from deep_control import utils as dc_utils

SUNRISE_AGENTS = {'sunrise', 'hypersunrise'}
TD3_AGENTS = {'td3', 'hypeRL_td3', 'polyL0_td3'}


# ==========================
#  EVALUATION
# ==========================

def eval_agent(agent, eval_env, eval_episodes, idx, name, best_rew, agent_type):
    avg_reward = 0.
    avg_state_cost = 0.
    avg_action_cost = 0.
    best_policy = False
    start_time = time.time()
    steps = 0

    for i in range(eval_episodes):
        episode_state_cost = 0.
        episode_action_cost = 0.
        state = eval_env.reset(i=i)
        done = False
        while not done:
            if agent_type in SUNRISE_AGENTS:
                action = agent.forward(state)
            else:
                action = agent.select_action(state)
            state, reward, done, info = eval_env.step(action)
            avg_reward += reward
            avg_state_cost += info["state_cost"]
            avg_action_cost += info["action_cost"]
            episode_state_cost += info["state_cost"]
            episode_action_cost += info["action_cost"]
            steps += 1

        try:
            eval_env.render(name=name, idx=idx, best_policy=best_policy,
                            agent_type=agent_type, i=i,
                            cost=[episode_state_cost, episode_action_cost])
        except Exception:
            pass

    avg_reward /= eval_episodes
    avg_state_cost /= eval_episodes
    avg_action_cost /= eval_episodes

    if avg_reward > best_rew:
        best_rew = avg_reward
        best_policy = True
        try:
            eval_env.render(name=name, idx=idx, best_policy=best_policy,
                            agent_type=agent_type,
                            cost=[avg_state_cost, avg_action_cost])
        except Exception:
            pass

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f} "
          f"steps: {steps} Time: {np.round(time.time() - start_time, 2)}")
    print("---------------------------------------")
    return avg_reward, best_rew, avg_state_cost, avg_action_cost


# ==========================
#  TD3-STYLE TRAINING
# ==========================

def train_td3(agent, env, eval_env, max_episodes, max_steps, warmup, eval_int,
              max_action, log, agent_type, save_int, discount,
              action_dim, batch_size, seed, replay_buffer):
    count = 0
    best_rew = -1e6
    for episode in range(max_episodes):
        start_time = time.time()
        state = env.reset()
        episode_reward = 0.0
        episode_state_cost = 0.0
        episode_action_cost = 0.0
        for step in range(max_steps):
            if episode < warmup:
                action = env.action_space.sample()
            else:
                action = (agent.select_action(state) +
                          np.random.normal(0, max_action * 0.1, size=action_dim)
                          ).clip(-max_action, max_action)

            next_state, reward, done, info = env.step(action)
            done_bool = 1 if done else 0
            replay_buffer.add(state, action, next_state, reward, done_bool)

            state = next_state
            episode_reward += reward
            episode_state_cost += info["state_cost"]
            episode_action_cost += info["action_cost"]
            count += 1

            if done:
                if count > batch_size:
                    agent.train(replay_buffer, iterations=100, batch_size=batch_size,
                                log=log, discount=discount)
                print(f"Episode Num: {episode + 1} Step: {step + 1} "
                      f"Reward: {episode_reward:.3f} "
                      f"Time: {np.round(time.time() - start_time, 2)}")
                if log:
                    wandb.log({'train/training_reward': episode_reward,
                               'train/training_state_cost': episode_state_cost,
                               'train/training_action_cost': episode_action_cost})

                if episode % eval_int == 0 and episode != 0:
                    eval_rew, best_rew, eval_sc, eval_ac = eval_agent(
                        agent=agent, eval_env=eval_env, eval_episodes=10,
                        idx=episode, name='testing', best_rew=best_rew,
                        agent_type=agent_type)
                    if log:
                        wandb.log({'eval/eval_reward': eval_rew,
                                   'eval/eval_state_cost': eval_sc,
                                   'eval/eval_action_cost': eval_ac})

                if episode % save_int == 0 and episode != 0:
                    directory = f'saved_models/{agent_type}seed_{seed}'
                    os.makedirs(directory, exist_ok=True)
                    agent.save(filename=f'/{agent_type}_{episode}', directory=directory)
                break

    directory = f'saved_models/{agent_type}seed_{seed}'
    os.makedirs(directory, exist_ok=True)
    agent.save(filename=f'/{agent_type}_last', directory=directory)

    try:
        eval_rew, _, _, _ = eval_agent(agent=agent, eval_env=eval_env, eval_episodes=10,
                                    idx=max_episodes + 1, name='testing',
                                    best_rew=-1e6, agent_type=agent_type)
        if log:
            wandb.log({'eval/eval_reward': eval_rew})
    except Exception as e:
        print(f"Warning: final evaluation failed ({e})")


# ==========================
#  SUNRISE TRAINING
# ==========================

def train_sunrise(agent, env, eval_env, max_episodes, max_steps, warmup, eval_int,
                  save_int, batch_size, tau, actor_lr, critic_lr, alpha_lr, gamma,
                  eval_episodes, target_delay, weighted_bellman_temp, init_alpha,
                  log, seed, agent_type, sunrise_buffer, learn_fn):

    save_dir = os.path.join("saved_models", agent_type, f"seed_{seed}")
    os.makedirs(save_dir, exist_ok=True)

    target_agent = copy.deepcopy(agent)
    for target_critic, agent_critic in zip(target_agent.critics, agent.critics):
        dc_utils.hard_update(target_critic, agent_critic)
    target_agent.train()

    critic_optimizer = torch.optim.Adam(
        chain(*(c.parameters() for c in agent.critics)),
        lr=critic_lr, betas=(0.9, 0.999))
    actor_optimizer = torch.optim.Adam(
        chain(*(a.parameters() for a in agent.actors)),
        lr=actor_lr, betas=(0.9, 0.999))

    log_alphas, alpha_optimizers = [], []
    for _ in range(len(agent.actors)):
        log_alpha = torch.Tensor([math.log(init_alpha)]).to(dc_utils.device)
        log_alpha.requires_grad = True
        alpha_optimizers.append(
            torch.optim.Adam([log_alpha], lr=alpha_lr, betas=(0.5, 0.999)))
        log_alphas.append(log_alpha)
    target_entropy = -env.action_space.shape[0]

    best_eval_return = -1e9

    for episode in range(max_episodes):
        state = env.reset()
        episode_reward = 0.0
        episode_state_cost = 0.0
        episode_action_cost = 0.0
        episode_start_time = time.time()

        for step in range(max_steps):
            if episode < warmup:
                action = env.action_space.sample()
            else:
                action = agent.sample_action(state)
            action = np.asarray(action, dtype=np.float32)

            next_state, reward, done, info = env.step(action)
            sunrise_buffer.push(state, action, reward, next_state, done)

            state = next_state
            episode_reward += reward
            episode_state_cost += info["state_cost"]
            episode_action_cost += info["action_cost"]

            if done:
                ep_time = time.time() - episode_start_time
                print(f"Episode Num: {episode + 1} Step: {step + 1} "
                      f"Reward: {episode_reward:.3f} Time: {ep_time:.2f}")
                if log:
                    wandb.log({'train/training_reward': episode_reward,
                               'train/training_state_cost': episode_state_cost,
                               'train/training_action_cost': episode_action_cost})

                if step > batch_size:
                    for up in range(100):
                        loss_info = learn_fn(
                            buffer=sunrise_buffer, target_agent=target_agent,
                            agent=agent, critic_optimizer=critic_optimizer,
                            batch_size=batch_size, gamma=gamma, critic_clip=None,
                            actor_optimizer=actor_optimizer,
                            alpha_optimizers=alpha_optimizers,
                            target_entropy=target_entropy, log_alphas=log_alphas,
                            actor_clip=None,
                            weighted_bellman_temp=weighted_bellman_temp)
                        if up % target_delay == 0:
                            for t_c, a_c in zip(target_agent.critics, agent.critics):
                                dc_utils.soft_update(t_c, a_c, tau)
                        if up % 2 == 0 and log:
                            wandb.log({'train/actor_loss': loss_info["actor_loss"],
                                       'train/critic_loss': loss_info["critic_loss"]})

                if episode % eval_int == 0:
                    print("\n=== EVALUATION ===")
                    eval_rew, best_eval_return, eval_sc, eval_ac = eval_agent(
                        agent=agent, eval_env=eval_env, eval_episodes=eval_episodes,
                        idx=episode, name='testing', best_rew=best_eval_return,
                        agent_type=agent_type)
                    if log:
                        wandb.log({'eval/eval_reward': eval_rew,
                                   'eval/eval_state_cost': eval_sc,
                                   'eval/eval_action_cost': eval_ac})

                if episode % save_int == 0 and episode != 0:
                    save_path = os.path.join(save_dir, f"{agent_type}_{episode}")
                    os.makedirs(save_path, exist_ok=True)
                    agent.save(save_path)
                break

    save_path = os.path.join(save_dir, f"{agent_type}_last")
    os.makedirs(save_path, exist_ok=True)
    agent.save(save_path)


# ==========================
#  MAIN
# ==========================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # --- Agent ---
    parser.add_argument('--agent-type', type=str, default='td3',
                        help='Agent type: td3, hypeRL_td3, polyL0_td3, sunrise, hypersunrise.')

    # --- Environment ---
    parser.add_argument('--T', type=int, default=300, help='Simulation time (seconds).')
    parser.add_argument('--Nx', type=int, default=64, help='Spatial discretization.')
    parser.add_argument('--Lx', type=int, default=22, help='Domain size.')
    parser.add_argument('--dt', type=float, default=0.1, help='Timestep PDE solver (seconds).')
    parser.add_argument('--frameskip', type=int, default=2, help='Control action repetition steps.')
    parser.add_argument('--control-start', type=int, default=100, help='Controller start time (seconds).')
    parser.add_argument('--oversampling', type=int, default=15, help='PDE solver oversampling.')
    parser.add_argument('--nr-actuators', type=int, default=8, help='Number of actuators.')
    parser.add_argument('--nr-sensors', type=int, default=8, help='Number of sensors.')
    parser.add_argument('--offset', type=int, default=4, help='Sensor/actuator offset.')
    parser.add_argument('--action-scale', type=float, default=1.0, help='Action scale factor.')
    parser.add_argument('--alpha', type=float, default=0.1, help='Reward action penalty weight.')
    parser.add_argument('--mu', type=float, default=0.0, help='System parameter.')
    parser.add_argument('--sigma', type=float, default=0.8, help='Gaussian actuator std.')
    parser.add_argument('--param-dim', type=int, default=2, help='Parameter dimension.')
    parser.add_argument('--parametric', type=bool, default=True, help='Randomize mu each episode.')
    parser.add_argument('--random-init', type=bool, default=True, help='Random initial conditions.')
    parser.add_argument('--random-target', type=bool, default=False, help='Random target state.')

    # --- Training ---
    parser.add_argument('--max-episodes', type=int, default=1000, help='Number of training episodes.')
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size.')
    parser.add_argument('--h-dim', type=int, default=256, help='Hidden layer size.')
    parser.add_argument('--warmup', type=int, default=50, help='Random policy warmup episodes.')
    parser.add_argument('--seed', type=int, default=1, help='Random seed.')
    parser.add_argument('--eval-int', type=int, default=200, help='Evaluation interval (episodes).')
    parser.add_argument('--eval-episodes', type=int, default=10, help='Evaluation episodes.')
    parser.add_argument('--save-int', type=int, default=1000, help='Model save interval (episodes).')
    parser.add_argument('--tau', type=float, default=0.005, help='Target network update rate.')
    parser.add_argument('--discount', type=float, default=0.99, help='Discount factor.')
    parser.add_argument('--param-repeat', type=bool, default=True, help='Repeat parameter in state.')

    # --- Sunrise-specific ---
    parser.add_argument('--ensemble-size', type=int, default=5, help='Ensemble size (sunrise).')
    parser.add_argument('--ucb-bonus', type=float, default=5.0, help='UCB bonus (sunrise).')
    parser.add_argument('--actor-lr', type=float, default=1e-4, help='Actor learning rate (sunrise).')
    parser.add_argument('--critic-lr', type=float, default=1e-4, help='Critic learning rate (sunrise).')
    parser.add_argument('--alpha-lr', type=float, default=1e-4, help='Alpha learning rate (sunrise).')
    parser.add_argument('--weighted-bellman-temp', type=float, default=20.0, help='Weighted Bellman temperature (sunrise).')
    parser.add_argument('--init-alpha', type=float, default=0.1, help='Initial entropy coefficient (sunrise).')
    parser.add_argument('--log-std-low', type=float, default=-10, help='Log std lower bound (sunrise).')
    parser.add_argument('--log-std-high', type=float, default=2, help='Log std upper bound (sunrise).')
    parser.add_argument('--target-delay', type=int, default=2, help='Target network update delay (sunrise).')
    parser.add_argument('--buffer-size', type=int, default=1_000_000, help='Replay buffer size (sunrise).')
    parser.add_argument('--prioritized-replay', action='store_true', help='Prioritized replay (sunrise).')

    # --- Logging ---
    parser.add_argument('--log', type=bool, default=True, help='Enable W&B logging.')

    args = parser.parse_args()

    # --- Derived parameters ---
    agent_type = args.agent_type
    max_episodes = args.max_episodes
    T = args.T
    Nx = args.Nx
    Lx = args.Lx
    dt = args.dt
    frameskip = args.frameskip
    control_start = args.control_start
    max_steps = int(((T / dt) - (control_start / dt)) / frameskip) - 1
    oversampling = args.oversampling
    batch_size = args.batch_size
    h_dim = args.h_dim
    warmup = args.warmup
    seed = args.seed
    eval_int = args.eval_int
    eval_episodes = args.eval_episodes
    nr_actuators = args.nr_actuators
    nr_sensors = args.nr_sensors
    action_scale = args.action_scale
    alpha = args.alpha
    tau = args.tau
    mu = args.mu
    sigma = args.sigma
    offset = args.offset
    param_dim = args.param_dim
    log = args.log
    parametric = args.parametric
    random_init = args.random_init
    random_target = args.random_target
    save_int = args.save_int
    param_repeat = args.param_repeat
    discount = args.discount

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env_name = 'KuramotoSivashinsky'

    # --- Environments ---
    env = KuramotoSivashinskyEnv(
        Nx=Nx, Lx=Lx, dt=dt, T=T, frameskip=frameskip, max_rl_steps=max_steps,
        parametric=parametric, random_target=random_target, action_scale=action_scale,
        nr_actuators=nr_actuators, nr_sensors=nr_sensors, oversampling=oversampling,
        alpha=alpha, mu=mu, sigma=sigma, offset=offset, control_start=control_start,
        seed=seed, eval=False, random_init=random_init)
    eval_env = KuramotoSivashinskyEnv(
        Nx=Nx, Lx=Lx, dt=dt, T=T, frameskip=frameskip, max_rl_steps=max_steps,
        parametric=parametric, random_target=random_target, action_scale=action_scale,
        nr_actuators=nr_actuators, nr_sensors=nr_sensors, oversampling=oversampling,
        alpha=alpha, mu=mu, sigma=sigma, offset=offset, control_start=control_start,
        seed=seed, eval=True, random_init=random_init)

    state_dim = env.observation_space.shape[0] + param_dim
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    env.action_space.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # --- W&B logging ---
    if log:
        config = {
            "env": env_name, "agent_type": agent_type,
            "max_episodes": max_episodes, "T": T, "Nx": Nx, "Lx": Lx, "dt": dt,
            "max_rl_steps": max_steps, "frameskip": frameskip,
            "oversampling": oversampling, "batch_size": batch_size,
            "hidden_dim": h_dim, "warmup": warmup, "seed": seed,
            "parametric": parametric, "eval_interval": eval_int,
            "nr_actuators": nr_actuators, "nr_sensors": nr_sensors,
            "action_scale": action_scale, "sigma": sigma, "alpha": alpha,
            "tau": tau, "mu": mu, "random_init": random_init,
            "param_repeat": param_repeat, "discount": discount,
        }
        wandb.init(
            project="hypEMBER",
            name=f"{agent_type}_ks_seed_{seed}",
            tags=[env_name, agent_type],
            config=config,
            settings={"_service_wait": 600, "init_timeout": 600})

    # --- Agent and buffer ---
    if agent_type == 'td3':
        agent = TD3(state_dim=state_dim, action_dim=action_dim, max_action=max_action,
                    h_dim=h_dim, tau=tau, device=device, param_repeat=param_repeat)
        replay_buffer = TrainingBuffer(state_dim=state_dim, action_dim=action_dim, device=device)

    elif agent_type == 'hypeRL_td3':
        agent = hypeRLTD3(state_dim=state_dim, action_dim=action_dim, max_action=max_action,
                          h_dim=h_dim, tau=tau, device=device, param_repeat=param_repeat)
        replay_buffer = TrainingBuffer(state_dim=state_dim, action_dim=action_dim, device=device)

    elif agent_type == 'polyL0_td3':
        agent = polyL0TD3(state_dim=state_dim, action_dim=action_dim, param_dim=param_dim,
                          max_action=max_action, h_dim=h_dim, tau=tau, device=device)
        replay_buffer = TrainingBuffer(state_dim=state_dim, action_dim=action_dim, device=device)

    elif agent_type == 'sunrise':
        obs_dim = state_dim
        agent = SunriseAgent(
            obs_space_size=obs_dim, act_space_size=action_dim,
            log_std_low=args.log_std_low, log_std_high=args.log_std_high,
            ensemble_size=args.ensemble_size, ucb_bonus=args.ucb_bonus)
        BufferClass = (replay.PrioritizedReplayBuffer
                       if args.prioritized_replay else replay.ReplayBuffer)
        sunrise_buffer = BufferClass(
            size=args.buffer_size, state_dtype=float,
            state_shape=(obs_dim,), action_shape=(action_dim,))
        replay_buffer = None  # not used by sunrise

    elif agent_type == 'hypersunrise':
        obs_dim = state_dim
        agent = HyperSunriseAgent(
            obs_space_size=obs_dim, act_space_size=action_dim,
            log_std_low=args.log_std_low, log_std_high=args.log_std_high,
            ensemble_size=args.ensemble_size, ucb_bonus=args.ucb_bonus)
        BufferClass = (replay.PrioritizedReplayBuffer
                       if args.prioritized_replay else replay.ReplayBuffer)
        sunrise_buffer = BufferClass(
            size=args.buffer_size, state_dtype=float,
            state_shape=(obs_dim,), action_shape=(action_dim,))
        replay_buffer = None  # not used by hypersunrise

    else:
        raise ValueError(f"Unknown agent type: {agent_type}. "
                         f"Choose from: {sorted(TD3_AGENTS | SUNRISE_AGENTS)}")


    # --- Training ---
    if agent_type in TD3_AGENTS:
        train_td3(agent=agent, env=env, eval_env=eval_env,
                  max_episodes=max_episodes, max_steps=max_steps, warmup=warmup,
                  eval_int=eval_int, max_action=max_action, log=log,
                  agent_type=agent_type, save_int=save_int, discount=discount,
                  action_dim=action_dim, batch_size=batch_size, seed=seed,
                  replay_buffer=replay_buffer)

    elif agent_type in SUNRISE_AGENTS:
        learn_fn = learn_sunrise if agent_type == 'sunrise' else learn_hypersunrise
        train_sunrise(agent=agent, env=env, eval_env=eval_env,
                      max_episodes=max_episodes, max_steps=max_steps, warmup=warmup,
                      eval_int=eval_int, save_int=save_int, batch_size=batch_size,
                      tau=tau, actor_lr=args.actor_lr, critic_lr=args.critic_lr,
                      alpha_lr=args.alpha_lr, gamma=discount,
                      eval_episodes=eval_episodes, target_delay=args.target_delay,
                      weighted_bellman_temp=args.weighted_bellman_temp,
                      init_alpha=args.init_alpha, log=log, seed=seed,
                      agent_type=agent_type, sunrise_buffer=sunrise_buffer,
                      learn_fn=learn_fn)

    if log:
        wandb.finish()
