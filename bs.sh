#!/bin/sh
#SBATCH --time=20:00:00
#SBATCH --partition=gpu
#SBATCH --qos=gpu-medium
#SBATCH --gres=gpu:rtx2080ti:1
echo $(hostname)
python script.py
