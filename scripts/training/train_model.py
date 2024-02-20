import argparse
import json
import os
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "gauge_id",
        type=str,
        help="The ID of the USGS gauge to use for the grid search.",
    )
    parser.add_argument(
        "-d",
        "--data_file",
        type=str,
        default="data/catchments_short.json",
        help="JSON file containing catchment definitions",
    )
    parser.add_argument(
        "-c",
        "--columns_file",
        type=str,
        default="data/columns.txt",
        help="Text file containing a list of column names",
    )
    parser.add_argument(
        "-e",
        "--epochs",
        type=int,
        default=50,
        help="Number of epochs to train Contributing Models",
    )
    parser.add_argument(
        "-t",
        "--train_stride",
        type=int,
        default=1,
        help="Stride for regression model training",
    )
    parser.add_argument(
        "-n",
        "--combiner_holdout_size",
        type=int,
        default=365 * 24 * 3,
        help="Number of points to train regression model on",
    )
    parser.add_argument(
        "-s",
        "--test_start",
        type=float,
        default=0.05,
        help="Starting point for backtesting",
    )
    parser.add_argument(
        "-b", "--test_stride", type=int, default=5, help="Stride for backtesting"
    )

    args = parser.parse_args()
    gauge_id = args.gauge_id
    data_file = args.data_file
    columns_file = args.columns_file
    epochs = args.epochs
    train_stride = args.train_stride
    combiner_holdout_size = args.combiner_holdout_size
    test_start = args.test_start
    test_stride = args.test_stride

    coordinates = get_coordinates_for_catchment(data_file, gauge_id)
    if coordinates is None:
        print(f"Unable to locate {gauge_id} in catchment data file.")
        exit(1)

    columns = get_columns(columns_file)
    dataset = get_training_data("all-weather-data", gauge_id, coordinates, columns)
    model = build_model_for_dataset(
        dataset, epochs, combiner_holdout_size, train_stride
    )

    root_dir = f"trained_models/RNN/Huber/{epochs}/"
    os.makedirs(root_dir, exist_ok=True)

    forecaster = TrainingForecaster(model, dataset, root_dir=root_dir)
    forecaster.fit()

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
