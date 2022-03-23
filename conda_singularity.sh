#!/bin/bash

#SBATCH --job-name=openmmlab_jupyter
#SBATCH --output=oml-%j.out
#SBATCH --error=oml-%j.err
#SBATCH --partition gpu
#SBATCH --ntasks-per-node=2
#SBATCH --nodes=1
#SBATCH --gres=gpu:1

module load anaconda3
module load cuda-10.2

source /opt/packages/anaconda3/etc/profile.d/conda.sh
conda activate openmmlabjp

jupyter lab --ip=0.0.0.0 --port=8888 --no-browser
