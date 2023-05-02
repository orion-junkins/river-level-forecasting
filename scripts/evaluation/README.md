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
This will train a model using the parameters from the provided JSON and append results as additional fields in the same json file. By default, only the centermost coordinate is used. Pass `--use_all_coords` to use all coordinates.

4) To run a job on SLURM, run `hpc_run_job.bash` or `dgx_run_job.bash`. The HPC script will run on Soundbendor nodes while the DGX script will run on DGX nodes. You may need to edit nodelist SBATCH arg in these files as needed. 

Use the same initial Model/Gauge Id arguments as `run_single_gs_job.py`. But, rather than passing a single job id, pass a starting and ending index. All jobs from the first index up to (but not including the second) will be run.
For example, to run the job stored in `0.json`, run:
```
sbatch scripts/evaluation/hpc_run_job.bash RNN 14219000 0 1
```
This will forward arguments to `run_single_gs_job.py` and run it on the SLURM node specified in `hpc_run_job.bash`. 

To run multiple jobs, ie `0.json`, `1.json`, and `2.json` run:
```
sbatch scripts/evaluation/hpc_run_job.bash RNN 14219000 0 3
```