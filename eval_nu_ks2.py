"""
eval_nu_ks2.py
--------------
Valuta le policy KS su envs/ks2.py per valori campionati di nu (viscosità).

Per ogni valore in NU_VALUES:
  - esegue N_SEEDS seed × EVAL_EPISODES episodi
  - registra mean/std del reward totale per agente

Agenti: TD3, polyL0, HyperL-TD3, Sunrise, HyperSunrise
        (stessi checkpoint di eval_noise_ks.py)

Nota: poiché ks2.reset() ricampiona nu internamente, usiamo un monkey-patch
locale a np.random.uniform per fissarlo durante il reset.
"""

import gc
import json
import os
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch

from envs.ks2 import KuramotoSivashinskyEnv
from agents.td3 import TD3
from agents.hypeRL_td3 import TD3 as hypeRLTD3
from agents.polyL0_td3 import TD3 as polyL0TD3
from deep_control.hypersunrise import SunriseAgent as HyperSunriseAgent
from deep_control.sunrise import SunriseAgent as SunriseAgent

# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------

NU_VALUES      = [1.0, 1.1, 1.25, 1.5, 1.75, 2.0]   # valori di nu da testare
EVAL_EPISODES  = 20
RESULTS_PATH   = "results_nu_ks2.json"

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
# ENVIRONMENT  (ks2: fisica con nu variabile)
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
    eval=False,
    random_init=True,
    nu=1.0,
)

state_dim  = NR_SENSORS + PARAM_DIM   # 10
action_dim = NR_ACTUATORS              # 8

# -----------------------------------------------------------------------------
# LOAD AGENTS   (identici a eval_noise_ks.py)
# -----------------------------------------------------------------------------

print("Loading agents ...")

td3_agent = TD3(
    state_dim=state_dim, action_dim=action_dim, max_action=1.0,
    h_dim=256, tau=0.005, device=DEVICE,
    param_dim=PARAM_DIM, param_repeat=True,
)
td3_agent.load(filename="td3_last", directory="save_models/ks/td3seed_0/td3seed_0")
print("  TD3 loaded")

polyl0_agent = polyL0TD3(
    state_dim=state_dim, action_dim=action_dim, max_action=1.0,
    h_dim=256, tau=0.005, device=DEVICE,
    param_dim=PARAM_DIM, degree_pi=3,
)
polyl0_agent.load(filename="polyL0_td3_last",
                  directory="save_models/ks/polyL0_td3seed_0/polyL0_td3seed_0")
print("  polyL0 loaded")

hyperl_agent = hypeRLTD3(
    state_dim=state_dim, action_dim=action_dim, max_action=1.0,
    h_dim=256, tau=0.005, device=DEVICE,
    param_dim=PARAM_DIM, param_repeat=True,
)
hyperl_agent.load(filename="/hypeRL_td3_last",
                  directory="save_models/ks/hypeRL_td3seed_0/hypeRL_td3seed_0")
print("  HyperL-TD3 loaded")

sunrise_agent = SunriseAgent(
    obs_space_size=state_dim, act_space_size=action_dim,
    log_std_low=-10, log_std_high=2, ensemble_size=5, ucb_bonus=5.0,
)
sunrise_agent.load("save_models/ks/sunrise_seed0/seed_0/sunrise_last")
print("  Sunrise loaded")

hypersunrise_agent = HyperSunriseAgent(
    obs_space_size=state_dim, act_space_size=action_dim,
    log_std_low=-10, log_std_high=2, ensemble_size=5, ucb_bonus=5.0,
)
hypersunrise_agent.load("save_models/ks/hypersunrise_seed0/seed_0/sunrise_last")
print("  HyperSunrise loaded")

# -----------------------------------------------------------------------------
# HELPER: reset con nu fissato
# -----------------------------------------------------------------------------

def reset_fixed_nu(nu_val: float):
    """
    Chiama env.reset() forzando self.nu = nu_val.
    ks2.reset() ricampiona nu con np.random.uniform; lo sostituiamo
    temporaneamente per restituire il valore desiderato.
    """
    _orig_uniform = np.random.uniform

    def _fixed_uniform(*args, **kwargs):
        return nu_val

    np.random.uniform = _fixed_uniform
    try:
        obs = env.reset()
    finally:
        np.random.uniform = _orig_uniform
    return obs

# -----------------------------------------------------------------------------
# EPISODE RUNNER
# -----------------------------------------------------------------------------

def run_episode(action_fn, nu_val: float) -> float:
    state = reset_fixed_nu(nu_val)
    done = False
    total_r = 0.0
    while not done:
        action = action_fn(state)
        state, reward, done, _ = env.step(action)
        total_r += reward
    return total_r

# -----------------------------------------------------------------------------
# EVALUATE
# -----------------------------------------------------------------------------

def evaluate(action_fn, agent_name: str) -> dict:
    """
    Valuta action_fn per ogni nu in NU_VALUES.
    Ritorna { nu: {"mean": float, "std": float} }
    """
    print(f"\n  {agent_name}")
    t0 = time.time()
    results = {}

    for nu in NU_VALUES:
        rewards = []
        for ep_idx in range(EVAL_EPISODES):
            r = run_episode(action_fn, nu)
            rewards.append(r)
            print(f"    nu={nu:.3f}  ep={ep_idx:3d}  r={r:+9.2f}  "
                  f"(running mean={float(np.mean(rewards)):+9.2f})", flush=True)

        torch.cuda.empty_cache()
        gc.collect()

        mean = float(np.mean(rewards))
        std  = float(np.std(rewards))
        results[nu] = {"mean": mean, "std": std}
        print(f"  --> nu={nu:.3f}  MEAN={mean:+9.2f}  STD={std:.2f}", flush=True)

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

print("\n" + "=" * 60)
print("  TEST: Nu robustness  (ks2, nu variabile, agente ignora nu)")
print("=" * 60)

all_results = {}
for key, fn, name in AGENTS:
    all_results[key] = evaluate(fn, name)

# -----------------------------------------------------------------------------
# STAMPA TABELLA RIASSUNTIVA
# -----------------------------------------------------------------------------

agent_keys = [k for k, _, _ in AGENTS]
col_w = 18

header = f"  {'nu':>6}  " + "".join(f"{k:>{col_w}}" for k in agent_keys)
sep    = "-" * len(header)
print(f"\n\n{sep}")
print("  Nu robustness — mean reward")
print(sep)
print(header)
print(sep)
for nu in NU_VALUES:
    row = f"  {nu:>6.3f}  "
    for k in agent_keys:
        m = all_results[k][nu]["mean"]
        row += f"{m:>{col_w}.2f}"
    print(row)
print(sep)

# -----------------------------------------------------------------------------
# SALVA JSON
# -----------------------------------------------------------------------------

serializable = {
    agent: {str(nu): v for nu, v in res.items()}
    for agent, res in all_results.items()
}
with open(RESULTS_PATH, "w") as f:
    json.dump(serializable, f, indent=2)
print(f"\nRisultati salvati -> {RESULTS_PATH}")
