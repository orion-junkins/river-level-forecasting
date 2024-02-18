# Interactive model training script. Useful for rapid experimentation. For more complex training, consider using the training script in src/rlf/forecasting/train.py.
#%%
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
    print(
        "Import error on rlf packages. Ensure rlf and its dependencies have been installed into the local environment."
    )
    print(e)
    exit(1)

# Whether or not to rebuild the dataset
rebuild_dataset = False

# Training Parameters
gauge_id = "12143400"
data_file = "data/catchments/12143400.json"
columns_file = "data/columns_ecmwf.txt"
epochs = 1
train_stride = 25
combiner_holdout_size = 365 * 24
test_start = 0.05
test_stride = 25

# Contributing Model Parameters (fast training, naive params)
model_variation = "Transformer"
use_future_covariates = False
model_params = {
    "random_state": 42,
    "input_chunk_length": 72,
    "output_chunk_length": 24,
    "d_model": 8,
    "nhead": 4,
    "num_encoder_layers" : 3,
    "num_decoder_layers" : 3,
    "dim_feedforward" : 32,
    "dropout": 0.1,
    "activation" : "relu",
    "batch_size": 64,
    "n_epochs": epochs,
    "force_reset": True,
    "pl_trainer_kwargs": {
        "accelerator": "cpu",
        "enable_progress_bar": True,  # be sure to disable if you use these params on HPC or output file is HUGE
    },
}


#%%
# Load the coordinates for which we will fetch weather data
coordinates = get_coordinates_for_catchment(data_file, gauge_id)
if coordinates is None:
    print(f"Unable to locate {gauge_id} in catchment data file.")
    exit(1)

#%%
# Load the columns (features) for which we will fetch weather data
columns = get_columns(columns_file)

#%%
if rebuild_dataset:
    dataset = get_training_data("ecmwf-weather-data", gauge_id, coordinates, columns)
    pickle.dump(dataset, open(f"data/dataset_{gauge_id}.pkl", "wb"))
else:
    dataset = pickle.load(open(f"data/dataset_{gauge_id}.pkl", "rb"))

#%%
# Build the model for the dataset
model = build_model_for_dataset(
    dataset, 
    epochs, 
    combiner_holdout_size,
    train_stride,model_variation=model_variation,
    contributing_model_kwargs=model_params
)

#%%
# Make a directory for storing this run
timestamp = datetime.now().strftime("%y-%m-%d_%H-%M")
id = str(np.random.randint(100000))
root_dir = f"trained_models/interactive_training/{epochs}/{timestamp}/{id}/"
os.makedirs(root_dir, exist_ok=True)

#%%
# Create a forecaster and fit the model
forecaster = TrainingForecaster(
    model, 
    dataset, 
    root_dir=root_dir, 
    use_future_covariates=use_future_covariates)
forecaster.fit()

#%%
# Backtest the model
contrib_test_errors = forecaster.backtest_contributing_models(
    start=test_start, stride=test_stride
)
contrib_val_errors = forecaster.backtest_contributing_models(
    run_on_validation=True, start=test_start, stride=test_stride
)
average_test_error = sum(contrib_test_errors) / len(contrib_test_errors)
average_val_error = sum(contrib_val_errors) / len(contrib_val_errors)

scores: Dict[str, Union[float, List[float]]] = {}
scores["contrib test errors"] = contrib_test_errors
scores["contrib val errors"] = contrib_val_errors
scores["contrib test error average"] = average_test_error
scores["contrib val error average"] = average_val_error

ensemble_test_error = forecaster.backtest(start=test_start, stride=test_stride)
ensemble_val_error = forecaster.backtest(
    run_on_validation=True, start=test_start, stride=test_stride
)

scores["ensemble test error"] = ensemble_test_error
scores["ensemble val error"] = ensemble_val_error

with open(f"{root_dir}{gauge_id}/scores.json", "w") as outfile:
    json.dump(scores, outfile)

# Print the results and finishing timestamp
print (f"Average test error: {average_test_error}")
print("Finished at: ", datetime.now().strftime("%y-%m-%d_%H-%M"))

# %%
