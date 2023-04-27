#!/bin/bash
MODEL=$1
GAUGE=$2
START=$3
END=$4
 
for ((i = $START; i < $END; i++)); do
    sbatch scripts/evaluation/hpc_run_single_job.bash $MODEL $GAUGE $i
done
