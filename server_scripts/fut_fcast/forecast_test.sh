#!/bin/bash
#SBATCH -J JUNKINSO_NBEATS_TRAINING     #job name
#SBATCH -p cascades
#SBATCH -A cascades
#SBATCH --gres=gpu:1		#use one gpu
#SBATCH --nodelist=cn-m-1
#SBATCH -t 4-00:00:00		#4 day timeout
#SBATCH --export=ALL
source ~/miniconda3/bin/activate darts		# activate env.
python forecast_test.py