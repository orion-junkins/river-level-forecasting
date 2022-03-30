#!/bin/bash
#SBATCH -J JUNKINSO_NBEATS_TRAINING     #job name
#SBATCH -p dgx2
#SBATCH -A cascades
#SBATCH --gres=gpu:1		#use one gpu
#SBATCH --nodelist=dgx2-5
#SBATCH -t 4-00:00:00		#4 day timeout
#SBATCH --export=ALL
source ~/miniconda3/bin/activate darts		# activate env.
python train_model.py $1 $2			# $1=model_save_dir, $2=model_index_to_train