# TEAM39_AGENT

**Team:** TEAM39AGENT  
**Authors:** Andy Ningan Zong (`nzong8@gatech.edu`), Alireza Moradi (`amoradi30@gatech.edu`)

## Description

PPO agent trained against the CEIA baseline using a dense competitive reward function with behavioral cloning (imitation) of the CEIA opponent. The agent learns to:

- Approach and push the ball toward the opponent goal
- Defend against the ball approaching its own net
- Maintain good team spacing with its teammate
- Mimic effective CEIA actions as a bootstrapping signal

## Files

- `agent.py` — `RayAgent` class that implements `AgentInterface`; auto-discovers checkpoint files
- `__init__.py` — exposes `RayAgent` for the environment's module loader
- `params.pkl` — pickled RLlib PPO training config (required for checkpoint restore)
- `params.json` — human-readable version of the config
- `checkpoint_000550/checkpoint-550` — trained model weights (~550 iterations vs CEIA)

## Usage

```bash
# Watch the agent play
python -m soccer_twos.watch -m TEAM39_AGENT

# Watch vs CEIA baseline
python -m soccer_twos.watch -m1 TEAM39_AGENT -m2 ceia_baseline_agent
```
