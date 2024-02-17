from datetime import datetime
import json
import statistics
import pandas as pd
from typing import Any, Dict, List, Optional, Sequence

from sklearn.linear_model import HuberRegressor

try:
    from darts.models.forecasting.forecasting_model import GlobalForecastingModel
    from darts.models.forecasting.linear_regression_model import RegressionModel
    from darts.models.forecasting.rnn_model import RNNModel
    from darts.models.forecasting.transformer_model import TransformerModel
    from darts.models.forecasting.nhits import NHiTSModel
    from darts.models.forecasting.nbeats import NBEATSModel
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
    from rlf.forecasting.data_fetching_utilities.level_provider.level_provider_nwis import LevelProviderNWIS
    from rlf.forecasting.data_fetching_utilities.weather_provider.aws_weather_provider import AWSWeatherProvider
    from rlf.forecasting.training_dataset import TrainingDataset
    from rlf.models.contributing_model import ContributingModel
    from rlf.models.ensemble import Ensemble
except ImportError as e:
    print("Import error on rlf packages. Ensure rlf and its dependencies have been installed into the local environment.")
    print(e)
    exit(1)

# Mapping of model variation names to their corresponding Darts classes.
MODEL_VARIATIONS = {
    "RNN": RNNModel,
    "Transformer": TransformerModel,
    "NBEATS": NBEATSModel,
    "NHiTS": NHiTSModel,
}

# Default parameters for the RNN model.
DEFAULT_RNN_PARAMS = {
    "input_chunk_length": 24,
    "random_state": 42,
    "training_length": 48,
    "batch_size": 64,
    "model": "GRU",
    "hidden_dim": 25,
    "n_rnn_layers": 1,
    "dropout": 0.00,
    "n_epochs": 50,
    "force_reset": True,
    "pl_trainer_kwargs": {
        "accelerator": "gpu",
        "enable_progress_bar": True,  # be sure to disable if you use these params on HPC or output file is HUGE
    },
    "optimizer_kwargs": {'lr': 0.00001}
}

# Larger RNNN model parameters
EXTENDED_RNN_PARAMS = {
    "input_chunk_length": 128,
    "random_state": 42,
    "training_length": 320,
    "batch_size": 32,
    "model": "GRU",
    "hidden_dim": 64,
    "n_rnn_layers": 4,
    "dropout": 0.01,
    "n_epochs": 50,
    "force_reset": True,
    "pl_trainer_kwargs": {
        "accelerator": "gpu",
        "enable_progress_bar": False,  # this stops the output file from being HUGE
    },
}

def get_columns(column_file: str) -> List[str]:
    """Get the list of columns from a text file.

    Args:
        column_file (str): path to the text file containing the columns.

    Returns:
        List[str]: List of columns to use
    """
    with open(column_file) as f:
        return [c.strip() for c in f.readlines()]


def get_center_coordinate(all_coords: List[Coordinate]) -> List[Coordinate]:
    """
    Reduce a list of coordinates to a single element list of only the centermost point.

    Args:
        all_coords (List[Coordinate]): The initial list of all coordinates.

    Returns:
        List[Coordinate]: A single element list containing only the centermost coordinate.
    """
    lats = []
    lons = []
    for coord in all_coords:
        lats.append(coord.lat)
        lons.append(coord.lon)

    median_lat = round((statistics.median(lats) / 0.1) * 0.1, 1)
    median_lon = round((statistics.median(lons) / 0.1) * 0.1, 1)

    coord = Coordinate(lon=median_lon, lat=median_lat)
    return ([coord])


def get_coordinates_for_catchment(filename: str, gauge_id: str, center_only: bool = False) -> Optional[List[Coordinate]]:
    """Get the list of coordinates for a specific gauge ID from a geojson file.

    Args:
        filename (str): geojson file that contains catchment information.
        gauge_id (str): gauge ID to retrieve coordinates for.
        center_only (bool, optional): Whether to return only the center coordinate or all coordinates. Defaults to False.

    Returns:
        Optional[List[Coordinate]]: List of coordinates for the given gauge or None if the gauge could not be found.
    """
    with open(filename) as f:
        target = json.load(f)

    for feature in target["features"]:
        if feature["properties"]["gauge_id"] == gauge_id:
            coordinates = [Coordinate(lon, lat) for lon, lat in feature["geometry"]["coordinates"]]
            if center_only:
                coordinates = get_center_coordinate(coordinates)
            return coordinates

    return None


def get_recent_available_timestamps(aws_dispatcher: AWSDispatcher, num_timestamps: int) -> List[str]:
    """Get a list of recent timestamps available in AWS.

    Args:
        aws_dispatcher (AWSDispatcher): AWSDispatcher to use to list files.
        num_timestamps (int): Number of timestamps to fetch.

    Returns:
        List[str]: List of timestamps available in AWS.
    """
    files = aws_dispatcher.list_files("current")
    timestamps = list(map(lambda x: x.split("/")[-1], files))
    timestamps = timestamps[-num_timestamps:]

    return timestamps


def get_level_true(starting_timestamps: List[str], inference_level_provider: LevelProviderNWIS, window_size: int) -> pd.DataFrame:
    """Get the true level for the specified gauge ID.

    Args:
        starting_timestamps (List[str]): List of starting timestamps
        inference_level_provider (LevelProviderNWIS): LevelProvider to use to fetch level data.
        window_size (int): Number of timesteps to fetch beyond the last starting timestamp.

    Returns:
        pd.DataFrame: DataFrame containing the level data. Data is datetime naive and in UTC.
    """
    # Find bounding timestamps
    dt_timestamps = [datetime.strptime(timestamp, "%y-%m-%d_%H-%M") for timestamp in starting_timestamps]
    start_date = min(dt_timestamps)
    end_date = max(dt_timestamps) + (window_size * pd.Timedelta("1 hour"))

    # Note that the level provider uses the date in the format "yyyy-mm-dd"
    start = datetime.strftime(start_date, "%Y-%m-%d")
    end = datetime.strftime(end_date, "%Y-%m-%d")
    level = inference_level_provider.fetch_level(start=start, end=end)
    level.rename(columns={"level": "level_true"}, inplace=True)
    level.index = level.index.tz_convert(None)

    return level


def get_training_data(
    base_path: str,
    gauge_id: str,
    coordinates: List[Coordinate],
    columns: List[str],
    rolling_sum_columns: Optional[List[str]] = None,
    rolling_mean_columns: Optional[List[str]] = None,
    rolling_window_sizes: Sequence[int] = (10 * 24, 30 * 24)
) -> TrainingDataset:
    """Generate the TrainingDataset for the given gauge ID, coordinates, and columns.

    Args:
        gauge_id (str): gauge ID this TrainingDataset will represent.
        coordinates (List[Coordinate]): List of coordinates for this TrainingDataset.
        columns (List[str]): Columns to request from the weather provider.
        rolling_sum_columns (Optional[List[str]], optional): Columns to generate rolling sums for. Defaults to None.
        rolling_mean_columns (Optional[List[str]], optional): Columns to generate rolling means for. Defaults to None.
        rolling_window_sizes (Sequence[int], optional): Window sizes to use for rolling sums and means. Defaults to (10 * 24, 30 * 24).

    Returns:
        TrainingDataset: A TrainingDataset instance for the specified gauge ID, coordinates and columns.
    """
    weather_provider = AWSWeatherProvider(
        coordinates,
        AWSDispatcher(base_path, "open-meteo")
    )
    level_provider = LevelProviderNWIS(gauge_id)
    catchment_data = CatchmentData(
        gauge_id,
        weather_provider,
        level_provider,
        columns=columns
    )
    dataset = TrainingDataset(catchment_data,
                              rolling_sum_columns=rolling_sum_columns,
                              rolling_mean_columns=rolling_mean_columns,
                              rolling_window_sizes=rolling_window_sizes)
    return dataset


def generate_base_contributing_model(num_epochs: int = 50, model_variation: str = "RNN", contributing_model_kwargs: Dict[str, Any] = DEFAULT_RNN_PARAMS) -> GlobalForecastingModel:
    """Generate a base model for an individual contributing model.

    Args:
        num_epochs (int): Number of epochs to train for. Defaults to 50.
        model_variation (str): Which model variation to use. Defaults to "RNN".
        contributing_model_kwargs (Dict[str, Any]): Keyword arguments to pass to the model constructor. Defaults to DEFAULT_RNN_PARAMS.

    Returns:
        GlobalForecastingModel: Base model.
    """

    if model_variation not in MODEL_VARIATIONS:
        raise ValueError(f"Model variation {model_variation} not found.")

    contributing_model_kwargs["n_epochs"] = num_epochs

    return MODEL_VARIATIONS[model_variation](**contributing_model_kwargs)


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
