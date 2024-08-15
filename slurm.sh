#!/bin/bash
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1

#! specify node
#SBATCH -w ngongotaha


srun poetry run python main.py 