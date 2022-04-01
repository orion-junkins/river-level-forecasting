#!/bin/bash
#SBATCH -J JUNKINSO_BLOCKRNN_TRAINING     #job name
#SBATCH -p dgx2
#SBATCH -A cascades
#SBATCH --gres=gpu:1		#use one gpu
#SBATCH --nodelist=dgx2-5
#SBATCH -t 6-00:00:00		
#SBATCH --export=ALL
source ~/miniconda3/bin/activate darts		# activate env.
python grid_search.py $1