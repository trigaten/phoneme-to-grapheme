#!/bin/sh
#SBATCH --partition=gpu
#SBATCH --qos=gpu-short
#SBATCH --gres=gpu:rtx2080ti:1
echo $(hostname)
python python.py
