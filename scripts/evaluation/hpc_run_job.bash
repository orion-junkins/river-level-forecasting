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

MODEL=$1
GAUGE=$2
START=$3
END=$4

echo SLURM Running single GS job. Model: $MODEL Gauge: $GAUGE Jobs $START up to but not including $END
source activate river-level-2
 
for ((i = $START; i < $END; i++)); do
    echo STARTING JOB $i
    python scripts/evaluation/run_single_gs_job.py $MODEL $GAUGE $i
done