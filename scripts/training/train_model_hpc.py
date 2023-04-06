import json
import sys
from typing import List, Optional

try:
    from darts.models.forecasting.forecasting_model import GlobalForecastingModel
    from darts.models.forecasting.regression_ensemble_model import RegressionEnsembleModel
    from darts.models.forecasting.rnn_model import RNNModel
except ImportError as e:
    print("Import error on darts packages. Ensure darts and its dependencies have been installed into the local environment.")
    print(e)
    exit(1)

try:
    from rlf.aws_dispatcher import AWSDispatcher
    from rlf.forecasting.catchment_data import CatchmentData
    from rlf.forecasting.data_fetching_utilities.coordinate import Coordinate
    from rlf.forecasting.data_fetching_utilities.level_provider.level_provider_nwis import LevelProviderNWIS
    from rlf.forecasting.data_fetching_utilities.weather_provider.aws_weather_provider import AWSWeatherProvider
    from rlf.forecasting.training_dataset import TrainingDataset
    from rlf.forecasting.training_forecaster import TrainingForecaster
    from rlf.models.contributing_model import ContributingModel
except ImportError as e:
    print("Import error on rlf packages. Ensure rlf and its dependencies have been installed into the local environment.")
    print(e)
    exit(1)


usage = """
    usage: train_model_hpc.py <gauge id> [-d <path/to/data/file.json>] [-e <epoch count>] [-c <path/to/column/file>]
        gauge id: The numeric gauge ID to train for. This will be used as the catchment name as well.
        -d <path/to/data/file.json>: The path to a geojson data file to use for catchment data.
                                     If not given then catchments_high_precision_short.json will be used.
        -e <epoch count>: The number of epochs to train the model for.
                          Must be an integer greater than 0.
                          If not given then a default number of 1 will be used.
        -c <path/to/column/file>: A path to a file that contains a list of columns to be used.
                                  The file must be plain text with a single column on each line.
                                  If not provided then a default list of 18 columns will be used.
                                  Check the python code itself for the default list.
"""


DEFAULT_EPOCHS = 1
DEFAULT_DATA_FILE = "catchments_short.json"
DEFAULT_COLUMNS = [
    "apparent_temperature",
    "cloudcover",
    "cloudcover_high",
    "cloudcover_low",
    "cloudcover_mid",
    "dewpoint_2m",
    "et0_fao_evapotranspiration",
    "precipitation",
    "pressure_msl",
    "rain",
    "relativehumidity_2m",
    "snowfall",
    "surface_pressure",
    "temperature_2m",
    "vapor_pressure_deficit",
    "winddirection_10m",
    "windgusts_10m",
    "windspeed_10m"
]


def generate_base_contributing_model(num_epochs: int) -> GlobalForecastingModel:
    """Generate a base model for an individual contributing model.

    Modify this function if you want to use a different base model besides RNN.

    Args:
        num_epochs (int): Number of epochs to train for.

    Returns:
        GlobalForecastingModel: Base model.
    """
    return RNNModel(
        96,
        n_epochs=num_epochs,
        force_reset=True,
        pl_trainer_kwargs={
            "accelerator": "gpu",
            "enable_progress_bar": False  # this stops the output file from being HUGE
        }
    )


def build_model_for_dataset(
    training_dataset: TrainingDataset,
    num_epochs: int
) -> RegressionEnsembleModel:
    """Build the EnsembleModel with the contributing models.

    Args:
        training_dataset (TrainingDataset): TrainingDataset instance that will be used to train the models.
        num_epochs (int): Number of epochs to train each contributing model.

    Returns:
        RegressionEnsembleModel: Built ensemble model.
    """
    model = RegressionEnsembleModel(
        [
            ContributingModel(generate_base_contributing_model(num_epochs), prefix)
            for prefix in training_dataset.subsets
        ],
        24*365*3
    )
    return model


def get_coordinates_for_catchment(filename: str, gauge_id: str) -> Optional[List[Coordinate]]:
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
            coordinates = [Coordinate(lon, lat) for lon, lat in feature["geometry"]["coordinates"]]
            return coordinates

    return None


def get_training_data(
    gauge_id: str,
    coordinates: List[Coordinate],
    columns: List[str]
) -> TrainingDataset:
    """Generate the TrainingDataset for the given gauge ID, coordinates, and columns.

    Args:
        gauge_id (str): gauge ID this TrainingDataset will represent.
        coordinates (List[Coordinate]): List of coordinates for this TrainingDataset.
        columns (List[str]): Columns to request from the weather provider.

    Returns:
        TrainingDataset: _description_
    """
    weather_provider = AWSWeatherProvider(
        coordinates,
        AWSDispatcher("all-weather-data", "open-meteo")
    )
    level_provider = LevelProviderNWIS(gauge_id)
    catchment_data = CatchmentData(
        gauge_id,
        weather_provider,
        level_provider,
        columns=columns
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


def get_parameters_from_args(args: List[str]) -> Optional[dict]:
    """Parse command line arguments to get the run parameters.

    It is assumed that args[0] contains the name of the script file.

    Args:
        args (List[str]): List of command line arguments. Usually gathered from sys.argv.

    Returns:
        Optional[dict]: dictionary of run parameters or None if an issue was found with parsing parameters
    """
    if len(args) < 2:
        return None

    run_parameters = {
        "data_file": "catchments_high_precision_short.json",
        "epochs": 1,
        "columns": None,
        "gauge_id": None,
    }

    i = 1

    while (i < len(args)):
        arg = args[i]
        if arg == "-d":
            i += 1
            if len(args) == i or args[i].startswith("-"):
                return None
            else:
                run_parameters["data_file"] = args[i]
        elif arg == "-e":
            i += 1
            if len(args) == i or args[i].startswith("-"):
                return None
            else:
                try:
                    run_parameters["epochs"] = int(args[i])
                except ValueError:
                    # couldn't parse the integer
                    return None
        elif arg == "-c":
            i += 1
            if len(args) == i or args[i].startswith("-"):
                return None
            else:
                run_parameters["columns"] = args[i]
        else:
            if run_parameters["gauge_id"]:
                return None
            run_parameters["gauge_id"] = arg
        i += 1

    if run_parameters["columns"] is not None:
        columns_file: str = str(run_parameters["columns"])
        run_parameters["columns"] = get_columns(columns_file)
    else:
        run_parameters["columns"] = DEFAULT_COLUMNS.copy()

    if run_parameters["gauge_id"] is None:
        return None

    return run_parameters


def main(args: List[str]) -> int:
    parameters = get_parameters_from_args(args)

    if parameters is None:
        print(usage)
        return 1

    coordinates = get_coordinates_for_catchment(parameters["data_file"], parameters["gauge_id"])

    if coordinates is None:
        print(f"Unable to locate {parameters['gauge_id']} in catchment data file.")
        return 1

    dataset = get_training_data(parameters["gauge_id"], coordinates, parameters["columns"])

    model = build_model_for_dataset(dataset, parameters["epochs"])

    print(f"Training Run Parameters: {parameters}")

    forecaster = TrainingForecaster(model, dataset)
    forecaster.fit()
    return 0


if __name__ == "__main__":
    exit(main(sys.argv))
