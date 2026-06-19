"""
eval_noise_gyro_vorticity.py
----------------------------
Same evaluation as eval_noise_gyro_new.py but for policies trained
with navigation_mode='vorticity' (obs: [gx-x, gy-y, vorticity, amp, freq], dim=5).

Policies from C:\\Users\\Utente\\Desktop\\gyro_finale\\vorticity (seed 0, last checkpoints).

TEST 1 — Observation noise (relative, first 3 obs dims: position error + vorticity)
TEST 2 — Parameter noise   (relative, amplitude + frequency)
"""

import gc
import json
import os
import time
from functools import partial

import numpy as np
import torch

from agents.hypeRL_td3 import TD3 as hypeRLTD3
from agents.polyL0_td3 import TD3 as polyL0TD3
from agents.td3 import TD3
from deep_control import nets as dc_nets
from deep_control.hypersunrise import SunriseAgent as HyperSunriseAgent
from deep_control.sunrise import SunriseAgent as SunriseAgent
from envs.gyro import Gyro

# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------

EVAL_EPISODES  = 100
NOISE_PERCENTS = [0, 5, 10, 20, 30, 40]

AMPLITUDE_RANGE = (0.05, 0.45)
FREQUENCY_RANGE = (0.5, 2 * np.pi / 3)
PARAM_CLAMP_MIN = 1e-3

# Maximum possible values for each observable (used to scale noise)
# obs[0] = gx - x    (relative x): domain x in [0, 2]
# obs[1] = gy - y    (relative y): domain y in [0, 1]
# obs[2] = vorticity: max |dv/dx - du/dy| for double gyre = pi^2 * intensity
_INTENSITY = 0.1
OBS_MAX = np.array([2.0, 1.0, np.pi**2 * _INTENSITY], dtype=np.float64)

# Maximum parameter values (upper bound of sampling ranges)
AMPLITUDE_MAX = AMPLITUDE_RANGE[1]   # 0.45
FREQUENCY_MAX = FREQUENCY_RANGE[1]   # 2*pi/3

EPISODE_SEED = 42
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_PATH = "results_noise_gyro_vorticity.json"

POLICIES_BASE = r"C:\Users\Utente\Desktop\gyro_finale\vorticity"

# -----------------------------------------------------------------------------
# ENVIRONMENT  (vorticity mode)
# -----------------------------------------------------------------------------

_common = dict(
    T=80, dt=0.1,
    parametric_target=True,
    parametric_gyro=True,
    random_init=False,
    seed=1,
    eval=True,
)

env_vor = Gyro(navigation_mode="vorticity", rel_position=True, **_common)

state_dim_vor = env_vor.observation_space.shape[0]   # 5
action_dim    = env_vor.action_space.shape[0]         # 2
max_action    = float(env_vor.action_space.high[0])
param_dim_vor = env_vor.param_dim                     # 2

print(f"env_vor: state_dim={state_dim_vor}  param_dim={param_dim_vor}")
print(f"action_dim={action_dim}  device={DEVICE}")

# -----------------------------------------------------------------------------
# SHARED EPISODE CONFIGS
# -----------------------------------------------------------------------------

def sample_episode_configs(n_episodes: int, seed: int) -> dict:
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
    x0        = cfg["x0"][ep]
    y0        = cfg["y0"][ep]
    gx        = cfg["gx"][ep]
    gy        = cfg["gy"][ep]
    amplitude = cfg["amplitude"][ep]
    frequency = cfg["frequency"][ep]
    x_phys    = 2.0 * x0

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

    # initial vorticity at t=0
    vor0 = eval_env.local_vorticity(x_phys, y0, t=0, nx=10, ny=10)  # shape (1,)
    eval_env.state = np.array([0.0, x_phys, y0, vor0[0]], dtype=np.float32)
    eval_env.ep_traj.append(eval_env.state.copy())

    obs = np.array([gx - x_phys, gy - y0, vor0[0]], dtype=np.float32)
    obs = np.concatenate([obs, eval_env.mu])   # 5D
    return obs

# -----------------------------------------------------------------------------
# TEST 1: OBSERVATION NOISE  (relative, first 3 components)
# -----------------------------------------------------------------------------

def run_episode_obs_noise(action_fn, eval_env, cfg, ep, noise_pct, noise_seed):
    rng   = np.random.RandomState(noise_seed)
    state = reset_env_from_config(eval_env, cfg, ep)
    done  = False
    total_reward = 0.0

    while not done:
        if noise_pct > 0.0:
            noisy_state = state.copy()
            for i in range(3):   # position error (2) + vorticity (1)
                std_i = noise_pct / 100.0 * OBS_MAX[i]
                noisy_state[i] += rng.normal(0.0, std_i)
            action = action_fn(noisy_state)
        else:
            action = action_fn(state)

        state, reward, terminated, truncated, _ = eval_env.step(action)
        done = terminated or truncated
        total_reward += reward

    return total_reward


def evaluate_obs_noise(action_fn, eval_env, agent_name, cfg,
                       eval_episodes=EVAL_EPISODES):
    print(f"\n{'-'*56}")
    print(f"  [obs noise]    {agent_name}")
    print(f"{'-'*56}")
    t0      = time.time()
    results = {}
    for pct in NOISE_PERCENTS:
        ep_rewards = [
            run_episode_obs_noise(action_fn, eval_env, cfg, ep, pct, noise_seed=ep)
            for ep in range(eval_episodes)
        ]
        torch.cuda.empty_cache(); gc.collect()
        mean = float(np.mean(ep_rewards))
        std  = float(np.std(ep_rewards))
        results[pct] = {"mean": mean, "std": std}
        print(f"  pct={pct:3d}%  ->  {mean:+9.3f}  ±  {std:.3f}")
    print(f"  done in {time.time()-t0:.1f}s")
    return results

# -----------------------------------------------------------------------------
# TEST 2: PARAMETER NOISE  (relative, amplitude + frequency)
# -----------------------------------------------------------------------------


def run_episode_param_noise(action_fn, eval_env, cfg, ep, noise_pct, noise_seed):
    rng            = np.random.RandomState(noise_seed)
    state          = reset_env_from_config(eval_env, cfg, ep)
    orig_amplitude = eval_env.amplitude
    orig_frequency = eval_env.frequency
    done           = False
    total_reward   = 0.0

    while not done:
        if noise_pct > 0.0:
            eval_env.amplitude = max(
                PARAM_CLAMP_MIN,
                orig_amplitude + rng.normal(0.0, noise_pct / 100.0 * AMPLITUDE_MAX),
            )
            eval_env.frequency = max(
                PARAM_CLAMP_MIN,
                orig_frequency + rng.normal(0.0, noise_pct / 100.0 * FREQUENCY_MAX),
            )
        action = action_fn(state)
        state, reward, terminated, truncated, _ = eval_env.step(action)
        done = terminated or truncated
        total_reward += reward

    eval_env.amplitude = orig_amplitude
    eval_env.frequency = orig_frequency
    eval_env.mu        = [orig_amplitude, orig_frequency]
    return total_reward


def evaluate_param_noise(action_fn, eval_env, agent_name, cfg,
                         eval_episodes=EVAL_EPISODES):
    print(f"\n{'-'*56}")
    print(f"  [param noise]  {agent_name}")
    print(f"{'-'*56}")
    t0      = time.time()
    results = {}
    for pct in NOISE_PERCENTS:
        ep_rewards = [
            run_episode_param_noise(action_fn, eval_env, cfg, ep, pct, noise_seed=ep)
            for ep in range(eval_episodes)
        ]
        torch.cuda.empty_cache(); gc.collect()
        mean = float(np.mean(ep_rewards))
        std  = float(np.std(ep_rewards))
        results[pct] = {"mean": mean, "std": std}
        print(f"  pct={pct:3d}%  ->  {mean:+9.3f}  ±  {std:.3f}")
    print(f"  done in {time.time()-t0:.1f}s")
    return results

# -----------------------------------------------------------------------------
# TEST 3: COMBINED  (obs noise + param noise, same pct)
# -----------------------------------------------------------------------------

def run_episode_combined(action_fn, eval_env, cfg, ep, noise_pct, noise_seed):
    rng_obs   = np.random.RandomState(noise_seed)
    rng_param = np.random.RandomState(noise_seed + 10000)

    state          = reset_env_from_config(eval_env, cfg, ep)
    orig_amplitude = eval_env.amplitude
    orig_frequency = eval_env.frequency
    done           = False
    total_reward   = 0.0

    while not done:
        noisy_state = state.copy()

        # obs noise — first 3 components (position error + vorticity)
        if noise_pct > 0.0:
            for i in range(3):
                std_i = noise_pct / 100.0 * OBS_MAX[i]
                noisy_state[i] += rng_obs.normal(0.0, std_i)

        # param noise — amplitude + frequency
        if noise_pct > 0.0:
            eval_env.amplitude = max(
                PARAM_CLAMP_MIN,
                orig_amplitude + rng_param.normal(0.0, noise_pct / 100.0 * AMPLITUDE_MAX),
            )
            eval_env.frequency = max(
                PARAM_CLAMP_MIN,
                orig_frequency + rng_param.normal(0.0, noise_pct / 100.0 * FREQUENCY_MAX),
            )

        action = action_fn(noisy_state)
        state, reward, terminated, truncated, _ = eval_env.step(action)
        done = terminated or truncated
        total_reward += reward

    eval_env.amplitude = orig_amplitude
    eval_env.frequency = orig_frequency
    eval_env.mu        = [orig_amplitude, orig_frequency]
    return total_reward


def evaluate_combined(action_fn, eval_env, agent_name, cfg,
                      eval_episodes=EVAL_EPISODES):
    print(f"\n{'-'*56}")
    print(f"  [combined]     {agent_name}")
    print(f"{'-'*56}")
    t0      = time.time()
    results = {}
    for pct in NOISE_PERCENTS:
        ep_rewards = [
            run_episode_combined(action_fn, eval_env, cfg, ep, pct, noise_seed=ep)
            for ep in range(eval_episodes)
        ]
        torch.cuda.empty_cache(); gc.collect()
        mean = float(np.mean(ep_rewards))
        std  = float(np.std(ep_rewards))
        results[pct] = {"mean": mean, "std": std}
        print(f"  pct={pct:3d}%  ->  {mean:+9.3f}  ±  {std:.3f}")
    print(f"  done in {time.time()-t0:.1f}s")
    return results

# -----------------------------------------------------------------------------
# LOAD AGENTS
# -----------------------------------------------------------------------------

print("\nLoading agents (vorticity, seed 0) ...")

# ── TD3 ──
_td3_dir = os.path.join(POLICIES_BASE, "td3_seed_0", "td3_seed_0")
td3_agent = TD3(
    state_dim=state_dim_vor, action_dim=action_dim, max_action=max_action,
    h_dim=256, tau=0.005, param_dim=param_dim_vor, param_repeat=True, device=DEVICE,
)
td3_agent.load(filename="td3_last", directory=_td3_dir)
print("  TD3 loaded")

# ── hypeRL-TD3 ──
_hyperl_dir = os.path.join(POLICIES_BASE, "hypeRL_td3_seed_0", "hypeRL_td3_seed_0")
hyperl_agent = hypeRLTD3(
    state_dim=state_dim_vor, action_dim=action_dim, param_dim=param_dim_vor,
    max_action=max_action, h_dim=256, tau=0.005, device=DEVICE,
    param_repeat=True,
)
hyperl_agent.actor.load_state_dict(
    torch.load(os.path.join(_hyperl_dir, "hypeRL_td3_last_actor"),
               map_location=DEVICE, weights_only=False)
)
print("  hypeRL-TD3 loaded")

# ── polyL0-TD3 ──
_polyl0_dir = os.path.join(POLICIES_BASE, "polyL0_td3_seed_0", "polyL0_td3_seed_0")
polyl0_agent = polyL0TD3(
    state_dim=state_dim_vor, action_dim=action_dim, max_action=max_action,
    param_dim=param_dim_vor, h_dim=256, degree_pi=3, device=DEVICE,
)
polyl0_agent.load(filename="polyL0_td3_last", directory=_polyl0_dir)
print("  polyL0-TD3 loaded")

# ── SUNRISE ──
_sunrise_dir = os.path.join(POLICIES_BASE, "sunrise_seed_0", "seed_0", "sunrise_last")
sunrise_agent = SunriseAgent(
    obs_space_size=state_dim_vor, act_space_size=action_dim,
    log_std_low=-10, log_std_high=2.0, ensemble_size=5, ucb_bonus=5.0,
)
sunrise_agent.load(_sunrise_dir)
print("  SUNRISE loaded")

# ── HyperSunrise ──
_hs_actor_cls     = partial(dc_nets.HyperStochasticActor, hidden_size=256)
_hypersunrise_dir = os.path.join(POLICIES_BASE, "hypersunrise_seed_0", "seed_0", "seed_0", "hypersunrise_last")
hypersunrise_agent = HyperSunriseAgent(
    obs_space_size=state_dim_vor, act_space_size=action_dim,
    log_std_low=-10, log_std_high=2.0, ensemble_size=5, ucb_bonus=5.0,
    actor_net_cls=_hs_actor_cls,
)
hypersunrise_agent.load(_hypersunrise_dir)
print("  HyperSunrise loaded")

# -----------------------------------------------------------------------------
# SHARED EPISODE CONFIGS
# -----------------------------------------------------------------------------

episode_cfg = sample_episode_configs(EVAL_EPISODES, EPISODE_SEED)
print(f"\nShared episode configs: seed={EPISODE_SEED}, n={EVAL_EPISODES}")

# -----------------------------------------------------------------------------
# RUN EVALUATIONS
# -----------------------------------------------------------------------------

AGENTS = [
    ("td3",          lambda s: td3_agent.select_action(s),       "TD3",          env_vor),
    ("hyperl",       lambda s: hyperl_agent.select_action(s),    "hypeRL-TD3",   env_vor),
    ("polyl0",       lambda s: polyl0_agent.select_action(s),    "polyL0-TD3",   env_vor),
    ("sunrise",      lambda s: sunrise_agent.forward(s),         "SUNRISE",      env_vor),
    ("hypersunrise", lambda s: hypersunrise_agent.forward(s),    "HyperSunrise", env_vor),
]

print("\n\n" + "="*56)
print("  TEST 1: OBSERVATION NOISE (relative, first 3 obs dims)")
print("="*56)
obs_results = {}
for key, action_fn, name, agent_env in AGENTS:
    obs_results[key] = evaluate_obs_noise(action_fn, agent_env, name, episode_cfg)

print("\n\n" + "="*56)
print("  TEST 2: PARAMETER NOISE (relative, per-parameter)")
print("="*56)
param_results = {}
for key, action_fn, name, agent_env in AGENTS:
    param_results[key] = evaluate_param_noise(action_fn, agent_env, name, episode_cfg)

print("\n\n" + "="*56)
print("  TEST 3: COMBINED (obs noise + param noise, same pct)")
print("="*56)
combined_results = {}
for key, action_fn, name, agent_env in AGENTS:
    combined_results[key] = evaluate_combined(action_fn, agent_env, name, episode_cfg)

# -----------------------------------------------------------------------------
# SUMMARY TABLES
# -----------------------------------------------------------------------------

agent_keys = [k for k, _, _, _ in AGENTS]
col_w      = 18

def print_table(title, results, levels, level_label):
    header = f"  {level_label:>6}  " + "".join(f"{k:>{col_w}}" for k in agent_keys)
    sep    = "-" * len(header)
    print(f"\n\n{sep}")
    print(f"  {title}")
    print(sep)
    print(header)
    print(sep)
    for lv in levels:
        row = f"  {lv:>6}  "
        for k in agent_keys:
            m = results[k][lv]["mean"]
            row += f"{m:>{col_w}.3f}"
        print(row)
    print(sep)

print_table("TEST 1 — Obs noise (pct% of each obs component)",
            obs_results, NOISE_PERCENTS, "pct%")
print_table("TEST 2 — Param noise (pct% of each parameter)",
            param_results, NOISE_PERCENTS, "pct%")
print_table("TEST 3 — Combined (obs + param, same pct%)",
            combined_results, NOISE_PERCENTS, "pct%")

# -----------------------------------------------------------------------------
# SAVE
# -----------------------------------------------------------------------------

combined = {
    "obs_noise": {
        agent: {str(k): v for k, v in res.items()}
        for agent, res in obs_results.items()
    },
    "param_noise": {
        agent: {str(k): v for k, v in res.items()}
        for agent, res in param_results.items()
    },
    "combined_noise": {
        agent: {str(k): v for k, v in res.items()}
        for agent, res in combined_results.items()
    },
}
with open(RESULTS_PATH, "w") as f:
    json.dump(combined, f, indent=2)
print(f"\nResults saved -> {RESULTS_PATH}")
