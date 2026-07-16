# Hypernetwork-Based Reinforcement Learning for Control of Parametrized Dynamical Systems

This repository contains the code for the paper *"Hypernetwork-Based Reinforcement Learning for Control of Parametrized Dynamical Systems"*.

We study the problem of controlling parametrized dynamical systems using deep reinforcement learning. The key idea is to use **hypernetworks** to generate the weights of a policy network conditioned on the system parameters, allowing a single agent to generalize across a family of systems without retraining.

## Environments

| Environment | Description |
|---|---|
| `KuramotoSivashinsky` | 1D Kuramoto-Sivashinsky PDE, controlled via actuators |
| `Gyro` | 2D gyrotactic microswimmer navigation in a fluid flow |

## Agents

| Agent | Description |
|---|---|
| `td3` | TD3 |
| `hypeRL_td3` | HypeRL: hypernetwork-based TD3 |
| `polyL0_td3` | TD3 with polynomial feature expansion (L0 regularization) |
| `sunrise` | SAC with ensemble |
| `hypEMBER` | SAC with ensemble and hypernetworks |


## Installation

```bash
# Clone the repository
git clone <repo-url>
cd <repo-folder>

# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows

# Install PyTorch (choose the right CUDA version from https://pytorch.org)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install remaining dependencies
pip install -r requirements.txt
```

## Usage

### Training

Train on the Kuramoto-Sivashinsky environment:

```bash
python train_ks.py --agent-type hypeRL_td3 --max-episodes 2000 --seed 1
```

Train on the Gyro environment:

```bash
python train_gyro.py --agent-type hypeRL_td3 --max-episodes 5000 --seed 1
```

### Key arguments (`train_ks.py` / `train_gyro.py`)

| Argument | Default | Description |
|---|---|---|
| `--agent-type` | `td3` | Agent to use |
| `--max-episodes` | `2000` | Number of training episodes |
| `--seed` | `1` | Random seed |
| `--parametric` | `True` | Randomize system parameter at each episode |
| `--log` | `False` | Enable W&B logging |
| `--h-dim` | `256` | Hidden layer size |

Run `python train_ks.py --help` for the full list of arguments.

## Repository Structure

```
├── agents/              # RL agents (TD3, D4PG, HypeRL, PolyL0)
│   └── nn/              # Neural network modules (hypernetworks, policies, Q-networks)
├── deep_control/        # SUNRISE and HyperSunrise implementations
├── envs/                # Custom environments (KS PDE, Gyro)
├── utils/               # Replay buffers and utilities
├── train_ks.py          # Training script — KS environment
├── train_gyro.py        # Training script — Gyro environment
├── sunrise_ks.py        # SUNRISE training — KS
├── sunrise_gyro.py      # SUNRISE training — Gyro
└── requirements.txt
```
