#!/bin/bash
#SBATCH -J junkinsTestGS     #job name
#SBATCH -p cascades
#SBATCH -A cascades
#SBATCH --gres=gpu:1		#use one gpu
#SBATCH --nodelist=cn-m-1	#use server 1
#SBATCH --mem=20G			#20GB RAM per node
#SBATCH -t 4-00:00:00		#4 day timeout
#SBATCH --mail-user=junkinso@oregonstate.edu
#SBATCH --mail-type=END
#SBATCH --export=ALL
echo SLURM Running single GS job. Model: $1 Gauge: $2 Job: $3
source activate river-level-2
python scripts/evaluation/run_single_gs_job.py $1 $2 $3 # RNN 14219000 0