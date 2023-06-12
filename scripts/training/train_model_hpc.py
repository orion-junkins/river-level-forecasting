import argparse
import json
import os
from sklearn.linear_model import HuberRegressor
from typing import Dict, List, Optional, Union

try:
    from darts.models.forecasting.forecasting_model import GlobalForecastingModel
    from darts.models.forecasting.linear_regression_model import RegressionModel
    from darts.models.forecasting.rnn_model import RNNModel
except ImportError as e:
    print(
        "Import error on darts packages. Ensure darts and its dependencies have been installed into the local environment."
    )
    print(e)
    exit(1)

try:
    from rlf.aws_dispatcher import AWSDispatcher
    from rlf.forecasting.catchment_data import CatchmentData
    from rlf.forecasting.data_fetching_utilities.coordinate import Coordinate
    from rlf.forecasting.data_fetching_utilities.level_provider.level_provider_nwis import (
        LevelProviderNWIS,
    )
    from rlf.forecasting.data_fetching_utilities.weather_provider.aws_weather_provider import (
        AWSWeatherProvider,
    )
    from rlf.forecasting.training_dataset import TrainingDataset
    from rlf.forecasting.training_forecaster import TrainingForecaster
    from rlf.models.contributing_model import ContributingModel
    from rlf.models.ensemble import Ensemble
except ImportError as e:
    print(
        "Import error on rlf packages. Ensure rlf and its dependencies have been installed into the local environment."
    )
    print(e)
    exit(1)


def generate_base_contributing_model(num_epochs: int) -> GlobalForecastingModel:
    """Generate a base model for an individual contributing model.

    Modify this function if you want to use a different base model besides RNN.

    Args:
        num_epochs (int): Number of epochs to train for.

    Returns:
        GlobalForecastingModel: Base model.
    """
    return RNNModel(
        input_chunk_length=128,
        random_state=42,
        training_length=320,
        batch_size=32,
        model="GRU",
        hidden_dim=64,
        n_rnn_layers=4,
        dropout=0.01,
        n_epochs=num_epochs,
        force_reset=True,
        pl_trainer_kwargs={
            "accelerator": "gpu",
            "enable_progress_bar": False,  # this stops the output file from being HUGE
        },
    )


def build_model_for_dataset(
    training_dataset: TrainingDataset,
    num_epochs: int,
    combiner_holdout_size: int,
    train_stride: int,
) -> Ensemble:
    """Build the EnsembleModel with the contributing models.

    Args:
        training_dataset (TrainingDataset): TrainingDataset instance that will be used to train the models.
        num_epochs (int): Number of epochs to train each contributing model.
        combiner_holdout_size (int): Number of steps to hold out for combiner model training.
        train_stride (int): Number of steps to stride when training the combiner model.

    Returns:
        Ensemble: Built ensemble model.
    """
    contributing_models = [
        ContributingModel(
            generate_base_contributing_model(num_epochs=num_epochs), prefix
        )
        for prefix in training_dataset.subsets
    ]

    regression_model = RegressionModel(
        lags=None, lags_future_covariates=[0], model=HuberRegressor()
    )

    model = Ensemble(
        regression_model,
        contributing_models,
        combiner_holdout_size=combiner_holdout_size,
        combiner_train_stride=train_stride,
    )

    return model


def get_coordinates_for_catchment(
    filename: str, gauge_id: str
) -> Optional[List[Coordinate]]:
    """Get the list of coordinates for a specific gauge ID from a geojson file.

    Args:
        filename (str): geojson file that contains catchment information.
        gauge_id (str): gauge ID to retrieve coordinates for.

    Returns:
        Optional[List[Coordinate]]: List of coordinates for the given gauge or None if the gauge could not be found.
    """
    with open(filename) as f:
        target = json.load(f)

    for feature in target["features"]:
        if feature["properties"]["gauge_id"] == gauge_id:
            coordinates = [
                Coordinate(lon, lat) for lon, lat in feature["geometry"]["coordinates"]
            ]
            return coordinates

    return None


def get_training_data(
    gauge_id: str, coordinates: List[Coordinate], columns: List[str]
) -> TrainingDataset:
    """Generate the TrainingDataset for the given gauge ID, coordinates, and columns.

    Args:
        gauge_id (str): gauge ID this TrainingDataset will represent.
        coordinates (List[Coordinate]): List of coordinates for this TrainingDataset.
        columns (List[str]): Columns to request from the weather provider.

    Returns:
        TrainingDataset: TrainingDataset instance for the given gauge ID, coordinates, and columns.
    """
    weather_provider = AWSWeatherProvider(
        coordinates, AWSDispatcher("all-weather-data", "open-meteo")
    )
    level_provider = LevelProviderNWIS(gauge_id)
    catchment_data = CatchmentData(
        gauge_id, weather_provider, level_provider, columns=columns
    )
    dataset = TrainingDataset(catchment_data)
    return dataset


def get_columns(column_file: str) -> List[str]:
    """Get the list of columns from a text file.

    Args:
        column_file (str): path to the text file containing the columns.

    Returns:
        List[str]: List of columns to use
    """
    with open(column_file) as f:
        return [c.strip() for c in f.readlines()]


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
        default=0,
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
    dataset = get_training_data(gauge_id, coordinates, columns)
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
