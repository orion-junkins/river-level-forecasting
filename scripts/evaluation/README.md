# RLF Evaluation tools
This folder contains a variety of helper scripts for evaluating model performance. 

## Grid Search
To run a grid search for a particular parameter space: 
1) Define the search space in json in `data/grid_search_[some search space].json`. `data/grid_search_0_model.json` is the default file for model parameter tuning.
2) Run `build_all_gs_jobs.py` with the desired model variation (must exist as a top level key in the search space json file) and the desired USGS gauge (must have a catchment definition at the expected location in `data/`).
For example, assuming there is an 'RNN' entry in the search space and assuming we have a catchment definition at the default location for gauge 14219000:
```
python scripts/evaluation/build_all_gs_jobs.py RNN 14219000
```
This will create a collection of grid search jobs at the path `grid_search/RNN/14219000/jobs`. 

In general, you will want to specify a search space file and an output directory. For example, to build jobs for the search space specified in `data/grid_search_1_lag_windows.json`:
```
python scripts/evaluation/build_all_gs_jobs.py RNN 14219000 -s data/grid_search_1_lag_windows.json -o grid_search_1_lag_windows
```
This will place jobs in the directory `grid_search_1_lag_windows/RNN/14219000/jobs`.

Pass `--help` to see additional options.


3) To run a single job, run `run_single_gs_job.py` with the model variation, gauge id and job id of the job you would like to run.
For example, to run the job stored in `0.json`, run:
```
python scripts/evaluation/run_single_gs_job.py RNN 14219000 0
```
This will train a model using the parameters from the provided JSON and append results as additional fields in the same json file. By default, only the centermost coordinate is used. Pass `--use_all_coords` to use all coordinates.

To specify an input dir, use the `-i` flag. For example, to run one of our `grid_search_1_lag_windows` jobs:
```
python scripts/evaluation/run_single_gs_job.py RNN 14219000 0 -i grid_search_1_lag_windows
```

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

To specify an specific input dir, simply pass one extra argument. The extra argument will be forwarded to the -i argument in the python script. For example, to run our `grid_search_1_lag_windows` jobs: 
```
sbatch scripts/evaluation/hpc_run_job.bash RNN 14219000 0 3 grid_search_1_lag_windows
```

5) To identify the best accuracies, you can use the `identify_best_score.py` script. It takes a jobs directory and prints the filename and average contributing test error for the lowest three scoring files.

For example:
```
python scripts/evaluation/identify_best_score.py grid_search/RNN/14377100/jobs
```