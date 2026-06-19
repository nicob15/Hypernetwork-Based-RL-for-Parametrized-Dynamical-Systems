"""
eval_noise_gyro.py
------------------
Two separate robustness tests on the Gyro environment (parametric target).

TEST 1 — Parameter (dynamics) noise
  At each step the physical parameters are perturbed additively:
    param_t = max(eps_min, param_nominal + sigma * N(0,1))
  The agent observes the NOMINAL mu throughout; only the physics change.
  sigma in PARAM_NOISE_SCALES = [0.1, 0.2, 0.3, 0.4]  (absolute)

TEST 2 — Observation noise
  Before passing the state to the agent, additive Gaussian noise is
  applied to the first 3 observation components.
  std in OBS_NOISE_STDS = [0.1, 0.2, 0.3, 0.4]  (absolute)

Agent observation formats (all agents — velocity, rel_position=True, 6D):
  [gx-x, gy-y, vx, vy, amplitude, frequency]
  TD3 uses param_repeat=True internally → actor sees 7D.

Shared episode configs
  Initial conditions (x0, y0), goal (gx, gy) and nominal parameters
  (amplitude, frequency) are pre-sampled ONCE with a fixed seed.
"""

import gc
import json
import os
import time
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import torch

from agents.hypeRL_td3 import TD3 as hypeRLTD3
from agents.polyL0_td3 import TD3 as polyL0TD3
from agents.td3 import TD3
from deep_control import nets as dc_nets
from deep_control.hypersunrise import SunriseAgent as HyperSunriseAgent
from envs.gyro import Gyro

# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------

EVAL_EPISODES = 100

PARAM_NOISE_SCALES = [0.1, 0.2, 0.3, 0.4]
OBS_NOISE_STDS     = [0.1, 0.2, 0.3, 0.4]

# Parameter ranges matching training distribution (envs/gyro.py reset)
AMPLITUDE_RANGE = (0.05, 0.45)                  # amplitude of gyro oscillation
FREQUENCY_RANGE = (0.5, 2 * np.pi / 3)         # angular frequency

PARAM_CLAMP_MIN = 1e-3

EPISODE_SEED = 42
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_PATH = "results_noise_gyro_param_target_seed1.json"

POLICIES_BASE = r"C:\Users\Utente\Desktop\gyro\policies_seed1"

# -----------------------------------------------------------------------------
# ENVIRONMENTS
# Two envs: one per observation format (training config).
# TD3     → naive, rel_position=False  (7D obs)
# others  → velocity, rel_position=True (6D obs)
# -----------------------------------------------------------------------------

_common = dict(
    T=80, dt=0.1,
    parametric_target=True,
    parametric_gyro=True,
    random_init=False,
    seed=1,
    eval=True,
)

env_vel = Gyro(navigation_mode="velocity", rel_position=True, **_common)

state_dim_vel = env_vel.observation_space.shape[0]   # 6
action_dim    = env_vel.action_space.shape[0]         # 2
max_action    = float(env_vel.action_space.high[0])
param_dim_vel = env_vel.param_dim                     # 2 (mu only, rel_pos=True)

print(f"env_vel   : state_dim={state_dim_vel}  param_dim={param_dim_vel}")
print(f"action_dim={action_dim}  device={DEVICE}")


# -----------------------------------------------------------------------------
# SHARED EPISODE CONFIGS
# -----------------------------------------------------------------------------

def sample_episode_configs(n_episodes: int, seed: int) -> dict:
    """Pre-sample IC, goal and nominal params, shared across all agents/noise levels."""
    rng = np.random.RandomState(seed)
    return {
        "x0":        rng.uniform(0.1, 0.9, n_episodes),
        "y0":        rng.uniform(0.1, 0.9, n_episodes),
        "gx":        rng.uniform(0.1, 1.9, n_episodes),
        "gy":        rng.uniform(0.1, 0.9, n_episodes),
        "amplitude": rng.uniform(*AMPLITUDE_RANGE, n_episodes),
        "frequency": rng.uniform(*FREQUENCY_RANGE, n_episodes),
    }


def reset_env_from_config(eval_env: Gyro, cfg: dict, ep: int) -> np.ndarray:
    """
    Deterministically reset eval_env from pre-sampled config[ep].
    Returns initial 6D observation: [gx-x, gy-y, vx, vy, amplitude, frequency]
    """
    x0        = cfg["x0"][ep]
    y0        = cfg["y0"][ep]
    gx        = cfg["gx"][ep]
    gy        = cfg["gy"][ep]
    amplitude = cfg["amplitude"][ep]
    frequency = cfg["frequency"][ep]
    x_phys    = 2.0 * x0      # physical x in [0, 2]

    # Common reset bookkeeping
    eval_env.target_reached    = False
    eval_env.ubtime            = 0
    eval_env.near_target       = 0
    eval_env.ep_target_reached = []
    eval_env.ep_traj           = []
    eval_env.ep_acts           = []

    eval_env.amplitude = amplitude
    eval_env.frequency = frequency
    eval_env.mu        = [amplitude, frequency]
    eval_env.goal      = [gx, gy]

    # velocity mode: state = [t=0, x, y, vx, vy]
    # obs (rel_position=True): [gx-x, gy-y, vx, vy, amp, freq]
    v0 = eval_env.doublegyreVEC(
        0.0, [x_phys, y0], [0.0, 0.0],
        eval_env.intensity, eval_env.amplitude, eval_env.frequency,
    )
    eval_env.state = np.array([0.0, x_phys, y0, v0[0], v0[1]], dtype=np.float32)
    eval_env.ep_traj.append(eval_env.state.copy())
    obs = np.array(
        [gx - x_phys, gy - y0, v0[0], v0[1]],
        dtype=np.float32,
    )
    obs = np.concatenate([obs, eval_env.mu])   # 6D

    return obs


# -----------------------------------------------------------------------------
# TEST 1: PARAMETER (DYNAMICS) NOISE
# -----------------------------------------------------------------------------

def run_episode_param_noise(
    action_fn,
    eval_env: Gyro,
    cfg: dict,
    ep: int,
    noise_scale: float,
    noise_seed: int,
) -> float:
    rng   = np.random.RandomState(noise_seed)
    state = reset_env_from_config(eval_env, cfg, ep)

    orig_amplitude = eval_env.amplitude
    orig_frequency = eval_env.frequency
    done           = False
    total_reward   = 0.0

    while not done:
        if noise_scale > 0.0:
            eval_env.amplitude = max(
                PARAM_CLAMP_MIN,
                orig_amplitude + noise_scale * rng.randn(),
            )
            eval_env.frequency = max(
                PARAM_CLAMP_MIN,
                orig_frequency + noise_scale * rng.randn(),
            )

        action = action_fn(state)
        state, reward, terminated, truncated, _ = eval_env.step(action)
        done = terminated or truncated
        total_reward += reward

    eval_env.amplitude = orig_amplitude
    eval_env.frequency = orig_frequency
    eval_env.mu        = [orig_amplitude, orig_frequency]

    plt.close("all")
    return total_reward


def evaluate_param_noise(
    action_fn,
    eval_env: Gyro,
    agent_name: str,
    cfg: dict,
    noise_scales: list = None,
    eval_episodes: int = EVAL_EPISODES,
) -> dict:
    if noise_scales is None:
        noise_scales = PARAM_NOISE_SCALES

    print(f"\n{'-' * 56}")
    print(f"  [param noise]  {agent_name}")
    print(f"{'-' * 56}")
    t0      = time.time()
    results = {}

    for scale in noise_scales:
        ep_rewards = [
            run_episode_param_noise(
                action_fn, eval_env, cfg, ep, scale,
                noise_seed=ep,
            )
            for ep in range(eval_episodes)
        ]
        torch.cuda.empty_cache()
        gc.collect()

        mean = float(np.mean(ep_rewards))
        std  = float(np.std(ep_rewards))
        results[scale] = {"mean": mean, "std": std}
        print(f"  sigma={scale:.2f}  ->  {mean:+9.3f}  ±  {std:.3f}")

    print(f"  done in {time.time() - t0:.1f}s")
    return results


# -----------------------------------------------------------------------------
# TEST 2: OBSERVATION NOISE
# -----------------------------------------------------------------------------

def run_episode_obs_noise(
    action_fn,
    eval_env: Gyro,
    cfg: dict,
    ep: int,
    obs_std: float,
    noise_seed: int,
) -> float:
    rng   = np.random.RandomState(noise_seed)
    state = reset_env_from_config(eval_env, cfg, ep)

    done         = False
    total_reward = 0.0

    while not done:
        if obs_std > 0.0:
            noisy_state       = state.copy()
            noisy_state[:3]  += obs_std * rng.randn(3)
            action = action_fn(noisy_state)
        else:
            action = action_fn(state)

        state, reward, terminated, truncated, _ = eval_env.step(action)
        done = terminated or truncated
        total_reward += reward

    plt.close("all")
    return total_reward


def evaluate_obs_noise(
    action_fn,
    eval_env: Gyro,
    agent_name: str,
    cfg: dict,
    obs_stds: list     = None,
    eval_episodes: int = EVAL_EPISODES,
) -> dict:
    if obs_stds is None:
        obs_stds = OBS_NOISE_STDS

    print(f"\n{'-' * 56}")
    print(f"  [obs noise]    {agent_name}")
    print(f"{'-' * 56}")
    t0      = time.time()
    results = {}

    for obs_std in obs_stds:
        ep_rewards = [
            run_episode_obs_noise(
                action_fn, eval_env, cfg, ep, obs_std,
                noise_seed=ep,
            )
            for ep in range(eval_episodes)
        ]
        torch.cuda.empty_cache()
        gc.collect()

        mean = float(np.mean(ep_rewards))
        std  = float(np.std(ep_rewards))
        results[obs_std] = {"mean": mean, "std": std}
        print(f"  std={obs_std:.2f}    ->  {mean:+9.3f}  ±  {std:.3f}")

    print(f"  done in {time.time() - t0:.1f}s")
    return results


# -----------------------------------------------------------------------------
# LOAD AGENTS
# -----------------------------------------------------------------------------

print("\nLoading agents (seed 1 policies) ...")

# TD3 — velocity env (6D obs), param_repeat=True
# Actor uses default param_dim=1; with state_dim=6 and param_repeat=True:
#   extended_dim = max(1, int(0.5*(6-1)//1)) = 2
#   effective input = (6-1) + 1*2 = 7  →  l1.weight [256,7]  ✓
_td3_dir = os.path.join(POLICIES_BASE, "td3_seed1")
td3_agent = TD3(
    state_dim=state_dim_vel, action_dim=action_dim, param_dim=param_dim_vel,
    max_action=max_action, h_dim=256, tau=0.005, device=DEVICE,
    param_repeat=True,
)
td3_agent.load(filename="td3_last", directory=_td3_dir)
print("  TD3 loaded")

# hypeRL — velocity env (6D obs), param_repeat=True
_hyperl_dir = os.path.join(POLICIES_BASE, "hyperl_seed1")
hyperl_agent = hypeRLTD3(
    state_dim=state_dim_vel, action_dim=action_dim, param_dim=param_dim_vel,
    max_action=max_action, h_dim=256, tau=0.005, device=DEVICE,
    param_repeat=True,
)
hyperl_agent.actor.load_state_dict(
    torch.load(os.path.join(_hyperl_dir, "hypeRL_td3_last_actor"), map_location=DEVICE)
)
hyperl_agent.critic.load_state_dict(
    torch.load(os.path.join(_hyperl_dir, "hypeRL_td3_last_critic"), map_location=DEVICE)
)
print("  hypeRL loaded")

# polyL0 — velocity env (6D obs)
_polyl0_dir = os.path.join(POLICIES_BASE, "poly_seed1")
polyl0_agent = polyL0TD3(
    state_dim=state_dim_vel, action_dim=action_dim, param_dim=param_dim_vel,
    max_action=max_action, h_dim=256, tau=0.005, device=DEVICE,
)
polyl0_agent.load(filename="polyL0_td3_last", directory=_polyl0_dir)
print("  polyL0 loaded")

# HyperSunrise — velocity env (6D obs), hidden_size=256
_hs_actor_cls = partial(dc_nets.HyperStochasticActor, hidden_size=256)
hypersunrise_agent = HyperSunriseAgent(
    obs_space_size=state_dim_vel, act_space_size=action_dim,
    log_std_low=-10, log_std_high=2.0, ensemble_size=5, ucb_bonus=5.0,
    actor_net_cls=_hs_actor_cls,
)
_hypersunrise_dir = os.path.join(POLICIES_BASE, "hypersunrise_seed17")
for _i in range(5):
    hypersunrise_agent.actors[_i].load_state_dict(
        torch.load(os.path.join(_hypersunrise_dir, f"actor{_i}.pt"), map_location=DEVICE)
    )
print("  HyperSunrise loaded (hypersunrise_seed17)")


# -----------------------------------------------------------------------------
# SHARED EPISODE CONFIGS
# -----------------------------------------------------------------------------

episode_cfg = sample_episode_configs(EVAL_EPISODES, EPISODE_SEED)

print(f"\nShared episode configs (seed={EPISODE_SEED}, n={EVAL_EPISODES}):")
print(f"  x0        in [{episode_cfg['x0'].min():.3f},  {episode_cfg['x0'].max():.3f}]")
print(f"  y0        in [{episode_cfg['y0'].min():.3f},  {episode_cfg['y0'].max():.3f}]")
print(f"  gx        in [{episode_cfg['gx'].min():.3f},  {episode_cfg['gx'].max():.3f}]")
print(f"  gy        in [{episode_cfg['gy'].min():.3f},  {episode_cfg['gy'].max():.3f}]")
print(f"  amplitude in [{episode_cfg['amplitude'].min():.3f},  {episode_cfg['amplitude'].max():.3f}]")
print(f"  frequency in [{episode_cfg['frequency'].min():.3f},  {episode_cfg['frequency'].max():.3f}]")


# -----------------------------------------------------------------------------
# RUN EVALUATIONS
# Each agent is paired with its native env (matching training config).
# -----------------------------------------------------------------------------

AGENTS = [
    ("td3",          lambda s: td3_agent.select_action(s),     "TD3",          env_vel),
    ("hyperl",       lambda s: hyperl_agent.select_action(s),  "hypeRL",       env_vel),
    ("polyl0",       lambda s: polyl0_agent.select_action(s),  "polyL0",       env_vel),
    ("hypersunrise", lambda s: hypersunrise_agent.forward(s),  "HyperSunrise", env_vel),
]

print("\n\n" + "=" * 56)
print("  TEST 1: PARAMETER (DYNAMICS) NOISE")
print("=" * 56)
param_results = {}
for key, action_fn, name, agent_env in AGENTS:
    param_results[key] = evaluate_param_noise(
        action_fn=action_fn, eval_env=agent_env, agent_name=name, cfg=episode_cfg,
    )

print("\n\n" + "=" * 56)
print("  TEST 2: OBSERVATION NOISE")
print("=" * 56)
obs_results = {}
for key, action_fn, name, agent_env in AGENTS:
    obs_results[key] = evaluate_obs_noise(
        action_fn=action_fn, eval_env=agent_env, agent_name=name, cfg=episode_cfg,
    )


# -----------------------------------------------------------------------------
# SUMMARY TABLES
# -----------------------------------------------------------------------------

agent_keys = [k for k, _, _, _ in AGENTS]
col_w      = 20


def print_table(title, results, levels, level_label):
    header = f"  {level_label:>6}  " + "".join(f"{k:>{col_w}}" for k in agent_keys)
    sep    = "-" * len(header)
    print(f"\n\n{sep}")
    print(f"  {title}")
    print(sep)
    print(header)
    print(sep)
    for lv in levels:
        row = f"  {lv:>6.2f}  "
        for k in agent_keys:
            m = results[k][lv]["mean"]
            row += f"{m:>{col_w}.3f}"
        print(row)
    print(sep)


print_table(
    "TEST 1 — Parameter noise (sigma, relative)",
    param_results, PARAM_NOISE_SCALES, "sigma",
)
print_table(
    "TEST 2 — Observation noise (std, absolute on first 3 obs dims)",
    obs_results, OBS_NOISE_STDS, "std",
)


# -----------------------------------------------------------------------------
# SAVE
# -----------------------------------------------------------------------------

combined = {
    "param_noise": {
        agent: {str(k): v for k, v in res.items()}
        for agent, res in param_results.items()
    },
    "obs_noise": {
        agent: {str(k): v for k, v in res.items()}
        for agent, res in obs_results.items()
    },
}
with open(RESULTS_PATH, "w") as f:
    json.dump(combined, f, indent=2)
print(f"\nResults saved -> {RESULTS_PATH}")
