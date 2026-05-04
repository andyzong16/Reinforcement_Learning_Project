#!/bin/bash
#SBATCH -A gts-phentenryck3-ai4opt
#SBATCH -J train_v2_export
#SBATCH -N 1 --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --output=train-ceia_long.out

# Optional: copy best-so-far (by episode_reward_mean) whenever a checkpoint is saved:
#   --live-best-dir "$SLURM_SUBMIT_DIR/TEAM2_AGENT_live"
python train_vs_ceia.py --timesteps 15000000 --workers 1 --export
