#!/bin/bash
#SBATCH -A gts-phentenryck3-ai4opt
#SBATCH -J train_v2_export
#SBATCH -N 1 --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=80G
#SBATCH --time=2:00:00
#SBATCH --output=train-v2.out

python train_strong_selfplay.py --timesteps 100000 --workers 1 --export
