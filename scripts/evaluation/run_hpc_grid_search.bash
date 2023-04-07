#!/bin/bash
#SBATCH -J junkinsTestGS     #job name
#SBATCH -p cascades
#SBATCH -A cascades
#SBATCH --gres=gpu:1		#use one gpu
#SBATCH --nodelist=cn-m-1	#use server 1
#SBATCH --mem=20G			#20GB RAM per node
#SBATCH -t 4-00:00:00		#4 day timeout
#SBATCH --export=ALL
source ~/miniconda3/bin/activate #ensure source activate works
source activate river-level		#activate GPU env.
python scripts/evaluation/build_grid_search_jobs.py RNN
python scripts/evaluation/run_grid_search_jobs.py RNN