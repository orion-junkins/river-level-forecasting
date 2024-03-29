#!/bin/bash
#SBATCH -w dgx2-3
#SBATCH -p dgx
#SBATCH -A eecs
#SBATCH --gres=gpu:1
#SBATCH --job-name=Junkins_GS
#SBATCH --mail-user=junkinso@oregonstate.edu
#SBATCH --mail-type=END
#SBATCH --export=ALL

MODEL=$1
GAUGE=$2
START=$3
END=$4
DIRECTORY=$5

echo SLURM Running single GS job. Model: $MODEL Gauge: $GAUGE Jobs $START up to but not including $END
source activate river-level-2
 
for ((i = $START; i < $END; i++)); do
    echo STARTING JOB $i
    if [ -z "$DIRECTORY" ]; then
        python scripts/evaluation/run_single_gs_job.py "$MODEL" "$GAUGE" "$i"
    else
        python scripts/evaluation/run_single_gs_job.py "$MODEL" "$GAUGE" "$i" -i "$DIRECTORY"
    fi
done