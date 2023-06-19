# %%
import os
from rlf.forecasting.data_fetching_utilities.weather_provider.api_weather_provider import APIWeatherProvider
from rlf.forecasting.training_forecaster import load_training_forecaster

import json
from typing import Dict, List, Union

try:
    from rlf.aws_dispatcher import AWSDispatcher
    from rlf.forecasting.catchment_data import CatchmentData
    from rlf.forecasting.data_fetching_utilities.level_provider.level_provider_nwis import LevelProviderNWIS
    from rlf.forecasting.data_fetching_utilities.weather_provider.aws_weather_provider import AWSWeatherProvider
    from rlf.forecasting.inference_forecaster import InferenceForecaster
    from rlf.forecasting.training_helpers import get_columns, get_coordinates_for_catchment
except ImportError as e:
    print("Import error on rlf packages. Ensure rlf and its dependencies have been installed into the local environment.")
    print(e)
    exit(1)


gauge_id = "14182500"
model_dir = "../trained_models/RNN/Huber/0"
data_file = f'../data/catchments/{gauge_id}.json'
columns_file = "../data/columns.txt"
output_dir = "tmp"

# Get columns
columns = get_columns(column_file=columns_file)

# Fetch coordinates for the specified gauge ID
coordinates = get_coordinates_for_catchment(data_file, gauge_id)
if coordinates is None:
    print("Unable to locate gauge id in catchment data file.")
    exit(1)

# Create AWSDispatcher and load available timestamps
aws_dispatcher = AWSDispatcher("all-weather-data", "open-meteo")

# Ceate weather and level providers for inference
inference_weather_provider = APIWeatherProvider(coordinates)
inference_level_provider = LevelProviderNWIS(gauge_id)
inference_catchment_data = CatchmentData(gauge_id, inference_weather_provider, inference_level_provider, columns=columns)
inference_forecaster = InferenceForecaster(inference_catchment_data, model_dir, load_cpu=False)

training_weather_provider = AWSWeatherProvider(coordinates, aws_dispatcher=aws_dispatcher)
tf = load_training_forecaster(inference_forecaster, training_weather_provider)

scores: Dict[str, Union[float, List[float]]] = {}
contrib_test_errors = tf.backtest_contributing_models(start=0.99, stride=100)
contrib_val_errors = tf.backtest_contributing_models(run_on_validation=True, start=0.99, stride=100)
average_test_error = sum(contrib_test_errors) / len(contrib_test_errors)
average_val_error = sum(contrib_val_errors) / len(contrib_val_errors)
scores["contrib test errors"] = contrib_test_errors
scores["contrib val errors"] = contrib_val_errors
scores["contrib test error average"] = average_test_error
scores["contrib val error average"] = average_val_error

ensemble_test_error = tf.backtest(start=0.99, stride=100)
ensemble_val_error = tf.backtest(run_on_validation=True, start=0.99, stride=100)

scores["ensemble test error"] = ensemble_test_error
scores["ensemble val error"] = ensemble_val_error

output_dir = os.path.join(output_dir, str(gauge_id))
os.makedirs(output_dir, exist_ok=True)
with open(f'{output_dir}/result.json', 'w') as fp:
    json.dump(scores, fp)
