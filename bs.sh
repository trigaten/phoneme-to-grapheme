#!/bin/sh
#SBATCH --partition=gpu
#SBATCH --time=6:00:00
#SBATCH --qos=gpu-medium
#SBATCH --gres=gpu:rtx2080ti:1
echo $(hostname)
python python.py
