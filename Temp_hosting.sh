#!/bin/bash -l

#SBATCH --output=/scratch/prj/inf_wqp/RQV/log.out
#SBATCH --job-name=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100

python SimpleUI.pyd