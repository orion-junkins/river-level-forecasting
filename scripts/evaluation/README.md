# RLF Evaluation tools
This folder contains a variety of helper scripts for evaluating model performance. 

## Grid Search
To run a grid search for a particular parameter space: 
1) Define the search space in json in 'data/grid_search_space.json'.
2) Run `build_grid_search_jobs.py` with the desired model variation (must exist as a top level key in the search space json file).
For example, assuming there is an 'RNN' entry in the search space:
```
python scripts/evaluation/build_grid_search_jobs.py RNN
```
Pass `--help` to see additional options.
3) To run a generated job, run `run_grid_search_jobs.py` with the model variation of the job you would like to run.
To run all jobs in the directory for our RNN search:
```
python scripts/evaluation/run_grid_search_jobs.py RNN
```
Individual jobs can be run with the -j flag. To run the 0th job for our RNN search:
```
python scripts/evaluation/run_grid_search_jobs.py RNN -j 0
```
