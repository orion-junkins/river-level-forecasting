# RLF Evaluation tools
This folder contains a variety of helper scripts for evaluating model performance. 

## Grid Search
To run a grid search for a particular parameter space: 
1) Define the search space in json in 'data/grid_search_space.json'.
2) Run `build_all_gs_jobs.py` with the desired model variation (must exist as a top level key in the search space json file) and the desired USGS gauge (must have a catchment definition at the expected location in `data/`).
For example, assuming there is an 'RNN' entry in the search space and assuming we have a catchment definition for gauge 14219000:
```
python scripts/evaluation/build_all_gs_jobs.py RNN 14219000
```
Pass `--help` to see additional options.

This will create a collection of grid search jobs at the path `grid_search/RNN/14219000/jobs`. 
3) To run a single job, run `run_single_gs_job.py` with the model variation, gauge id and job id of the job you would like to run.
For example, to run the job stored in `0.json`, run:
```
python scripts/evaluation/run_single_gs_job.py RNN 14219000 0
```
This will train a model using the parameters from the provided JSON and append results as additional fields in the same json file.

4) To run a single job on SLURM, run `hpc_run_single_job.bash` with the same arguments as `run_single_gs_job.py`
For example, to run the job stored in `0.json`, run:
```
sbatch scripts/evaluation/hpc_run_single_job.bash RNN 14219000 0
```
This will simply forward arguments to `run_single_gs_job.py` and run it on the SLURM node specified in `hpc_run_single_job.bash`. 
5) To run multiple jobs on SLURM, run `hpc_run_multiple_jobs.bash`. Use the same arguments as `run_single_gs_job.py`except instead of providing a single job id, provide a minimum and a maximum. All jobs with ids greater than or equal to the lower bound and less than the upper bound will be run. 
For example, to run the jobs stored in `0.json` and `1.json`, run:
```
sbatch scripts/evaluation/hpc_run_multiple_jobs.bash RNN 14219000 0 2
```