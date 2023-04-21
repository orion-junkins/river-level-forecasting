#!/bin/bash
MODEL=$1
GAUGE=$2
START=$3
END=$4
 
for i in {$START..$END}
do
    bash scripts/evaluation/hpc_run_single_job.bash $1 $2 $i
done
