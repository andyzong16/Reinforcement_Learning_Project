# Soccer-Twos — TEAM39 Agent

**Team Name:** TEAM39AGENT  
**Members:** Andy Ningan Zong (`nzong8@gatech.edu`) · Alireza Moradi (`amoradi30@gatech.edu`)  
**Course:** Georgia Tech — College of Computing  
**Competition:** CoRL 2026 Soccer-Twos Reinforcement Learning Challenge

---

## Overview

This repository contains the full training pipeline and a pre-trained agent for the [Soccer-Twos](https://github.com/bryanoliveira/soccer-twos-env) environment. Our agent uses **PPO** with a custom dense reward function that includes:

- **Ball progress reward** — incentivizes pushing the ball toward the opponent's goal
- **Approach & touch reward** — rewards moving toward and making contact with the ball
- **Defense shaping** — penalizes letting the ball drift dangerously close to our own net
- **Team spacing reward** — keeps the two blue-team agents spread apart
- **CEIA imitation reward** — behavioral cloning signal that mimics the baseline agent's actions

Our final model was trained against the CEIA baseline and beats it, a random agent, and the TA agent **at least 9/10 times**.

---

## Repository Structure

```
soccer-twos-TEAM39/
│
├── TEAM39_AGENT/               # Pre-trained agent (submit as zip for grading)
│   ├── __init__.py
│   ├── agent.py                # RayAgent — loads checkpoint and serves actions
│   ├── params.pkl              # Saved RLlib training config
│   ├── params.json             # Human-readable config snapshot
│   └── checkpoint_000550/
│       └── checkpoint-550      # Best checkpoint (~550 training iterations vs CEIA)
│
├── train_vs_ceia.py            # ★ PRIMARY training script (PPO vs CEIA with imitation)
├── train_agent.py              # Self-play PPO with archived-opponent rotation
├── train_strong_selfplay.py    # Strong self-play baseline
├── utils.py                    # CompetitiveRewardWrapper + RLlib env factory
│
├── example_random_players.py   # Quick sanity check — two random agents
├── example_random_teams.py     # Random team mode sanity check
│
├── train_ceia.slurm            # SLURM job script (PACE / shared HPC)
├── train_v2_export.slurm       # SLURM script for long 15M-step run
├── curriculum.yaml             # Curriculum config (optional advanced training)
├── requirements.txt            # Pinned Python dependencies
└── README.md                   # This file
```

---

## Setup

> **Python 3.8 is required.** The pinned dependencies (`ray==1.4.0`, `mlagents==0.27.0`) are not compatible with newer Python versions.

### 1 — Clone the repo

```bash
git clone https://github.com/<your-username>/soccer-twos-TEAM39.git
cd soccer-twos-TEAM39
```

### 2 — Create a conda environment

```bash
conda create --name soccertwos python=3.8 -y
conda activate soccertwos
```

### 3 — Downgrade build tools (required for old packages)

```bash
pip install pip==23.3.2 setuptools==65.5.0 wheel==0.38.4
pip cache purge
```

### 4 — Install dependencies

```bash
pip install -r requirements.txt
```

### 5 — Fix protobuf / pydantic compatibility

```bash
pip install protobuf==3.20.3
pip install pydantic==1.10.13
```

### 6 — Install the CEIA baseline agent

Download the pre-trained baseline checkpoint from the course Google Drive link, then extract it so you have:

```
soccer-twos-TEAM39/
└── ceia_baseline_agent/
    └── ray_results/
        └── PPO_selfplay_twos/
            └── PPO_Soccer_f475e_00000_0_2021-09-19_15-54-02/
                └── checkpoint_002449/
                    └── checkpoint-2449
```

> The `ceia_baseline_agent/` folder is git-ignored because it is large (~hundreds of MB).

---

## Quick Start — Watch a Random Game

```bash
python example_random_players.py
```

---

## Training TEAM39's Agent

### Option A — Train vs CEIA baseline (recommended, our best strategy)

This is the script that produced the checkpoint in `TEAM39_AGENT/`.

```bash
python train_vs_ceia.py --timesteps 5000000 --export
```

| Flag | Default | Description |
|------|---------|-------------|
| `--timesteps` | `5_000_000` | Total environment steps |
| `--workers` | `1` | Number of RLlib rollout workers |
| `--export` | off | Copy best checkpoint into `TEAM_AGENT/` at end |
| `--export-dir` | `./TEAM_AGENT` | Where to export the final agent |
| `--base-port` | auto | Unity port (change to avoid clashes on shared nodes) |
| `--live-best-dir` | off | Continuously export best checkpoint during training |
| `--ceia-checkpoint` | auto-detected | Override path to CEIA checkpoint file |

**Recommended full run (matches our paper results):**

```bash
python train_vs_ceia.py \
    --timesteps 15000000 \
    --workers 1 \
    --export \
    --export-dir ./TEAM39_AGENT
```

Training time is roughly **4–12 hours** on a CPU-only machine (more workers = faster). Use the SLURM script on PACE:

```bash
sbatch train_ceia.slurm
```

### Option B — Self-play with opponent archive rotation

```bash
python train_agent.py
```

This trains the agent against a rotating archive of past versions of itself. Useful for curriculum bootstrapping before switching to vs-CEIA fine-tuning.

### Option C — Strong self-play

```bash
python train_strong_selfplay.py
```

---

## Key Hyperparameters (PPO)

These are set in `train_vs_ceia.py` and were tuned for our best results:

| Parameter | Value | Why |
|-----------|-------|-----|
| `lr` | `3e-4` | Standard Adam learning rate |
| `gamma` | `0.99` | High discount — rewards long-term positioning |
| `lambda` | `0.95` | GAE smoothing reduces variance in policy gradients |
| `clip_param` | `0.2` | PPO clipping prevents catastrophic policy updates |
| `entropy_coeff` | `0.008` | Encourages exploration vs. strong fixed opponent |
| `vf_loss_coeff` | `3.0` | Extra emphasis on value head (shared layers) |
| `grad_clip` | `0.5` | Gradient clipping prevents exploding gradients |
| `fcnet_hiddens` | `[512, 512]` | Two-layer MLP policy/value network |

---

## Reward Function

Defined in `utils.py` (`CompetitiveRewardWrapper`) and configured in `train_vs_ceia.py` (`REWARD_CONFIG`):

| Component | Coefficient | Description |
|-----------|-------------|-------------|
| `ball_progress_coef` | `0.10` | Ball moving toward opponent goal |
| `approach_ball_coef` | `0.025` | Agent moving toward ball |
| `touch_ball_coef` | `0.014` | Agent touching the ball |
| `behind_ball_coef` | `0.012` | Agent positioned behind ball in attack direction |
| `defense_shape_coef` | `0.016` | Defensive positioning reward |
| `own_goal_danger_coef` | `0.025` | Penalty when ball is near own net |
| `spacing_coef` | `0.005` | Team spacing — avoid bunching |
| `step_penalty` | `0.0005` | Small per-step penalty encourages efficiency |
| `imitation_coef` | `0.035` | Match CEIA's action (behavioral cloning signal) |

---

## Evaluating the Agent

Watch the pre-trained `TEAM39_AGENT` play:

```bash
python -m soccer_twos.watch -m TEAM39_AGENT
```

Watch TEAM39 vs the CEIA baseline:

```bash
python -m soccer_twos.watch -m1 TEAM39_AGENT -m2 ceia_baseline_agent
```

Watch TEAM39 vs random:

```bash
python -m soccer_twos.watch -m1 TEAM39_AGENT -m2 example_player_agent
```

---

## Submitting the Agent

1. Make sure `TEAM39_AGENT/` contains:
   - `__init__.py`
   - `agent.py`
   - `params.pkl`
   - `checkpoint_*/checkpoint-*` (the trained weights)

2. Zip the folder:

```bash
zip -r TEAM39_AGENT.zip TEAM39_AGENT/
```

3. Submit `TEAM39_AGENT.zip` to the course portal.

---

## Results

Our agent achieves:
- ✅ Beats **random agent** ≥ 9/10 games  
- ✅ Beats **CEIA baseline** ≥ 9/10 games  
- ✅ Beats **TA agent** ≥ 9/10 games  

Mean reward climbs from **−2.0 → +0.75** over ~1200 training iterations (see Figure 1 in the paper).

---

## References

1. B. Oliveira. [soccer-twos-env](https://github.com/bryanoliveira/soccer-twos-env), 2021.  
2. K. Kurach et al. "Google Research Football: A Novel Reinforcement Learning Environment." AAAI 2020. [arXiv:1907.11180](https://arxiv.org/abs/1907.11180)

---

## Consent

We consent to the instruction team using our Soccer-Twos submission (code, models, and report) for research and publication purposes. We understand this is voluntary, does not impact our course grades, and our models will be anonymized.
