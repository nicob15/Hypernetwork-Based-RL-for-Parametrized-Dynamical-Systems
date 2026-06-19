"""
eval_noise_ks.py
----------------
Two robustness tests on KS environment with RELATIVE noise.

TEST 1 — Observation (sensor) noise
  At each RL step compute std(sensor_readings), then add
  N(0, x/100 * std) to each sensor component.
  x in NOISE_PERCENTS = [0, 5, 10, 20, 30, 40]

TEST 2 — Parameter misspecification
  Agent sees  mu_obs = mu + N(0, x/100 * |mu|)  while physics uses real mu.
  x in NOISE_PERCENTS = [0, 5, 10, 20, 30, 40]

Agents: TD3 (seed 51), polyL0 (seed 51), HyperL-TD3 (seed 69),
        Sunrise (seed 51), HyperSunrise (seed 1)
"""

import gc
import json
import os
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch

from envs.ks import KuramotoSivashinskyEnv
from agents.td3 import TD3
from agents.hypeRL_td3 import TD3 as hypeRLTD3
from agents.polyL0_td3 import TD3 as polyL0TD3
from deep_control.hypersunrise import SunriseAgent as HyperSunriseAgent
from deep_control.sunrise import SunriseAgent as SunriseAgent

# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------

N_SEEDS        = 5     # training seeds for IC diversity
EVAL_EPISODES  = 20   # episodes per seed
NOISE_PERCENTS = [0, 5, 10, 20, 30, 40]   # x% values
RESULTS_PATH   = "results_noise_ks.json"

T             = 300
NX            = 64
LX            = 22
DT            = 0.1
FRAMESKIP     = 2
CONTROL_START = 100
OVERSAMPLING  = 15
NR_ACTUATORS  = 8
NR_SENSORS    = 8
ACTION_SCALE  = 1.0
ALPHA         = 0.1
SIGMA         = 0.8
OFFSET        = 4
PARAM_DIM     = 2
MAX_RL_STEPS  = int(((T / DT) - (CONTROL_START / DT)) / FRAMESKIP) - 1

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------------------------------------------------------
# ENVIRONMENT
# -----------------------------------------------------------------------------

env = KuramotoSivashinskyEnv(
    Nx=NX, Lx=LX, dt=DT, T=T, frameskip=FRAMESKIP,
    max_rl_steps=MAX_RL_STEPS,
    parametric=True,
    random_target=False,
    action_scale=ACTION_SCALE,
    nr_actuators=NR_ACTUATORS,
    nr_sensors=NR_SENSORS,
    oversampling=OVERSAMPLING,
    alpha=ALPHA,
    mu=0.0,
    sigma=SIGMA,
    offset=OFFSET,
    control_start=CONTROL_START,
    seed=1,
    eval=True,
    random_init=True,
)

state_dim  = NR_SENSORS + PARAM_DIM   # 10
action_dim = NR_ACTUATORS              # 8

# -----------------------------------------------------------------------------
# LOAD AGENTS
# -----------------------------------------------------------------------------

print("Loading agents ...")

# TD3 seed 0
td3_agent = TD3(
    state_dim=state_dim, action_dim=action_dim, max_action=1.0,
    h_dim=256, tau=0.005, device=DEVICE,
    param_dim=PARAM_DIM, param_repeat=True,
)
td3_agent.load(filename="td3_last", directory="save_models/ks/td3seed_0/td3seed_0")
print("  TD3 loaded (seed 0)")

# polyL0 seed 0
polyl0_agent = polyL0TD3(
    state_dim=state_dim, action_dim=action_dim, max_action=1.0,
    h_dim=256, tau=0.005, device=DEVICE,
    param_dim=PARAM_DIM, degree_pi=3,
)
polyl0_agent.load(filename="polyL0_td3_last", directory="save_models/ks/polyL0_td3seed_0/polyL0_td3seed_0")
print("  polyL0 loaded (seed 0)")

# HyperL-TD3 seed 0
hyperl_agent = hypeRLTD3(
    state_dim=state_dim, action_dim=action_dim, max_action=1.0,
    h_dim=256, tau=0.005, device=DEVICE,
    param_dim=PARAM_DIM, param_repeat=True,
)
hyperl_agent.load(filename="/hypeRL_td3_last", directory="save_models/ks/hypeRL_td3seed_0/hypeRL_td3seed_0")
print("  HyperL-TD3 loaded (seed 0)")

# Sunrise seed 0
sunrise_agent = SunriseAgent(
    obs_space_size=state_dim, act_space_size=action_dim,
    log_std_low=-10, log_std_high=2, ensemble_size=5, ucb_bonus=5.0,
)
sunrise_agent.load("save_models/ks/sunrise_seed0/seed_0/sunrise_last")
print("  Sunrise loaded (seed 0)")

# HyperSunrise seed 0
hypersunrise_agent = HyperSunriseAgent(
    obs_space_size=state_dim, act_space_size=action_dim,
    log_std_low=-10, log_std_high=2, ensemble_size=5, ucb_bonus=5.0,
)
hypersunrise_agent.load("save_models/ks/hypersunrise_seed0/seed_0/sunrise_last")
print("  HyperSunrise loaded (seed 0)")

# -----------------------------------------------------------------------------
# EPISODE RUNNER
# -----------------------------------------------------------------------------

def run_episode(action_fn, obs_noise_pct=0.0, param_noise_pct=0.0, noise_rng=None):
    """
    Run one episode.
    obs_noise_pct  : x such that sensor noise std = x/100 * range(sensors), applied each step
    param_noise_pct: x such that param error std  = x/100 * range(param_vector), sampled ONCE
    """
    state   = env.reset()
    done    = False
    total_r = 0.0

    # --- parameter misspecification: sample error ONCE per episode ---
    param_error = 0.0
    if param_noise_pct > 0.0:
        param_range = float(np.max(state[0, NR_SENSORS:]) - np.min(state[0, NR_SENSORS:]))
        param_std   = param_noise_pct / 100.0 * param_range
        if param_std > 0.0:
            param_error = noise_rng.normal(0.0, param_std)

    while not done:
        s = state.copy()

        # --- sensor noise ---
        if obs_noise_pct > 0.0:
            sensor_std = float(np.max(s[0, :NR_SENSORS]) - np.min(s[0, :NR_SENSORS]))
            noise_std  = obs_noise_pct / 100.0 * sensor_std
            if noise_std > 0.0:
                s[0, :NR_SENSORS] += noise_rng.normal(0.0, noise_std, NR_SENSORS)

        # --- apply fixed param error ---
        if param_error != 0.0:
            s[0, NR_SENSORS] += param_error

        action = action_fn(s)
        state, reward, done, _ = env.step(action)
        total_r += reward

    return total_r


# -----------------------------------------------------------------------------
# EVALUATE ONE AGENT OVER ALL NOISE LEVELS
# -----------------------------------------------------------------------------

def evaluate(action_fn, agent_name, test):
    """
    test: "obs" or "param"
    N_SEEDS seeds × EVAL_EPISODES episodes each.
    Returns dict { pct: {"mean": float, "std": float} }
    """
    print(f"\n  [{test} noise]  {agent_name}")
    t0      = time.time()
    results = {}

    for pct in NOISE_PERCENTS:
        rewards = []
        for seed in range(N_SEEDS):
            np.random.seed(seed)
            torch.manual_seed(seed)
            noise_rng = np.random.RandomState(seed + 5000)
            for _ in range(EVAL_EPISODES):
                if test == "obs":
                    r = run_episode(action_fn, obs_noise_pct=pct,
                                    param_noise_pct=0.0, noise_rng=noise_rng)
                elif test == "param":
                    r = run_episode(action_fn, obs_noise_pct=0.0,
                                    param_noise_pct=pct, noise_rng=noise_rng)
                else:  # combined
                    r = run_episode(action_fn, obs_noise_pct=pct,
                                    param_noise_pct=pct, noise_rng=noise_rng)
                rewards.append(r)

        torch.cuda.empty_cache()
        gc.collect()
        mean = float(np.mean(rewards))
        std  = float(np.std(rewards))
        results[pct] = {"mean": mean, "std": std}
        print(f"    x={pct:2d}%  ->  {mean:+9.2f}  ±  {std:.2f}", flush=True)

    print(f"  done in {time.time() - t0:.1f}s")
    return results


# -----------------------------------------------------------------------------
# AGENTS LIST
# -----------------------------------------------------------------------------

AGENTS = [
    ("td3",          lambda s: td3_agent.select_action(s),        "TD3"),
    ("polyl0",       lambda s: polyl0_agent.select_action(s),     "polyL0"),
    ("hyperl",       lambda s: hyperl_agent.select_action(s),     "HyperL-TD3"),
    ("sunrise",      lambda s: sunrise_agent.forward(s),          "Sunrise"),
    ("hypersunrise", lambda s: hypersunrise_agent.forward(s),     "HyperSunrise"),
]

# -----------------------------------------------------------------------------
# RUN
# -----------------------------------------------------------------------------

# Carica risultati test 1 e 2 già esistenti
with open(RESULTS_PATH) as f:
    _existing = json.load(f)
obs_results   = {agent: {int(k): v for k, v in res.items()} for agent, res in _existing["obs_noise"].items()}
param_results = {agent: {int(k): v for k, v in res.items()} for agent, res in _existing["param_noise"].items()}

print("\n" + "=" * 60)
print("  TEST 3: COMBINED  (obs noise + param error, same x%)")
print("=" * 60)
combined_results = {}
for key, fn, name in AGENTS:
    combined_results[key] = evaluate(fn, name, test="combined")

# -----------------------------------------------------------------------------
# SUMMARY
# -----------------------------------------------------------------------------

agent_keys = [k for k, _, _ in AGENTS]
col_w = 16

def print_table(title, results, label):
    cell_w = col_w * 2 + 3   # "mean ± std" cell width
    header = f"  {'x%':>4}  {'':6}" + "".join(f"{k:>{cell_w}}" for k in agent_keys)
    sep    = "-" * len(header)
    print(f"\n\n{sep}")
    print(f"  {title}")
    print(sep)
    print(header)
    print(sep)
    for pct in NOISE_PERCENTS:
        row_mean = f"  {pct:>4d}  {'mean':>6}"
        row_std  = f"  {'':>4}  {'std':>6}"
        for k in agent_keys:
            m = results[k][pct]['mean']
            s = results[k][pct]['std']
            row_mean += f"{m:>{cell_w}.2f}"
            row_std  += f"{s:>{cell_w}.2f}"
        print(row_mean)
        print(row_std)
        print()
    print(sep)

print_table("TEST 1 — Obs noise (x% sensor std)",          obs_results,      "x%")
print_table("TEST 2 — Param noise (x% of |mu|)",          param_results,    "x%")
print_table("TEST 3 — Combined (obs + param, same x%)",   combined_results, "x%")

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
