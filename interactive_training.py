# Interactive model training script. Useful for rapid experimentation. For more complex training, consider using the training script in src/rlf/forecasting/train.py.
# %%
import json
import os
from datetime import datetime
import numpy as np
import pickle
from typing import Dict, List, Union

try:
    from rlf.forecasting.training_forecaster import TrainingForecaster
    from rlf.forecasting.training_helpers import (
        get_columns,
        get_coordinates_for_catchment,
        get_training_data,
        build_model_for_dataset,
    )
except ImportError as e:
    print("Import error on rlf packages. Ensure rlf and its dependencies have been installed into the local environment.")
    print(e)
    exit(1)

# Whether or not to rebuild the dataset
rebuild_dataset = False

# Training Parameters
gauge_id = "12143400"
data_file = "data/catchments/12143400.json"  # Note: Ignored if rebuild_dataset is False
columns_file = "data/columns_ecmwf.txt"  # Note: Ignored if rebuild_dataset is False
aws_bucket = "ecmwf-weather-data"  # Note: Ignored if rebuild_dataset is False
epochs = 0
train_stride = 25
combiner_holdout_size = 365 * 24

# Contributing Model Parameters (fast training, naive params)
model_variation = "Transformer"
use_future_covariates = False
model_params = {
    "random_state": 42,
    "input_chunk_length": 72,
    "output_chunk_length": 24,
    "d_model": 8,
    "nhead": 4,
    "num_encoder_layers": 3,
    "num_decoder_layers": 3,
    "dim_feedforward": 32,
    "dropout": 0.1,
    "activation": "relu",
    "batch_size": 64,
    "n_epochs": epochs,
    "force_reset": True,
    "pl_trainer_kwargs": {
        "accelerator": "cpu",  # Set to GPU if available
        "enable_progress_bar": True,  # Disable on HPC or output file is HUGE
    },
}

# Testing Parameters
forecast_horizon = 24
test_start = 0.05
test_stride = 25

# %% Load or build Dataset
if rebuild_dataset:
    coordinates = get_coordinates_for_catchment(data_file, gauge_id)
    if not coordinates:
        print(f"Coordinates not found for gauge {gauge_id}. Exiting.")
        exit(1)
    columns = get_columns(columns_file)
    dataset = get_training_data(aws_bucket, gauge_id, coordinates, columns)
    pickle.dump(dataset, open(f"data/dataset_{gauge_id}.pkl", "wb"))
else:
    dataset = pickle.load(open(f"data/dataset_{gauge_id}.pkl", "rb"))

# %% Build the model
model = build_model_for_dataset(
    dataset,
    epochs,
    combiner_holdout_size,
    train_stride,
    model_variation=model_variation,
    contributing_model_kwargs=model_params
)

# %% Make a directory for storing this run
timestamp = datetime.now().strftime("%y-%m-%d_%H-%M")
id = str(np.random.randint(10000))
root_dir = f"trained_models/interactive_training/{model_variation}/{epochs}/{timestamp}_{id}/"
os.makedirs(root_dir, exist_ok=True)

# %%
# Create a forecaster and fit the model
forecaster = TrainingForecaster(
    model,
    dataset,
    root_dir=root_dir,
    use_future_covariates=use_future_covariates)
forecaster.fit()

# %%
# Backtest contributing models
contrib_test_errors = forecaster.backtest_contributing_models(
    start=test_start,
    forecast_horizon=forecast_horizon,
    stride=test_stride,
)
contrib_val_errors = forecaster.backtest_contributing_models(
    run_on_validation=True,
    start=test_start,
    forecast_horizon=forecast_horizon,
    stride=test_stride,
)

average_test_error = sum(contrib_test_errors) / len(contrib_test_errors)
average_val_error = sum(contrib_val_errors) / len(contrib_val_errors)

scores: Dict[str, Union[float, List[float]]] = {}
scores["contrib test errors"] = contrib_test_errors
scores["contrib val errors"] = contrib_val_errors
scores["contrib test error average"] = average_test_error
scores["contrib val error average"] = average_val_error

# %% Backtest entire Ensemble
ensemble_test_error = forecaster.backtest(
    start=test_start,
    stride=test_stride,
    forecast_horizon=forecast_horizon
)
ensemble_val_error = forecaster.backtest(
    run_on_validation=True,
    start=test_start,
    stride=test_stride,
    forecast_horizon=forecast_horizon
)

scores["ensemble test error"] = ensemble_test_error
scores["ensemble val error"] = ensemble_val_error

# Write scores to json file
with open(f"{root_dir}/scores.json", "w") as f:
    json.dump(scores, f, indent=4)

# Write training parameters to json file
with open(f"{root_dir}/training_params.json", "w") as f:
    json.dump({
        "gauge_id": gauge_id,
        "data_file": data_file,
        "columns_file": columns_file,
        "aws_bucket": aws_bucket,
        "epochs": epochs,
        "train_stride": train_stride,
        "combiner_holdout_size": combiner_holdout_size,
        "use_future_covariates": use_future_covariates,
        "forecast_horizon": forecast_horizon,
        "test_start": test_start,
        "test_stride": test_stride,
        "model_variation": model_variation,
        "model_params": model_params,
    }, f, indent=4)

# Print the results and finishing timestamp
print(f"Average test error: {average_test_error}")
print("Finished at: ", datetime.now().strftime("%y-%m-%d_%H-%M"))
