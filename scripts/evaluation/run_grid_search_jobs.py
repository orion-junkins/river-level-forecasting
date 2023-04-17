import argparse
import json
import os
from typing import Any, List, Optional

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

# Universal job parameters expected regardless of model variation. All other parameters will be passed as kwargs to the contributing model.
STANDARD_JOB_PARAMETERS = ["data_file", "columns_file", "gauge_id", "model_variation", "regression_train_n_points"]

# Mapping of model variation names to their corresponding Darts classes.
MODEL_VARIATIONS = {
    "RNN": RNNModel
}


def generate_base_contributing_model(model_variation: str, contributing_model_kwargs: dict[str, Any]) -> GlobalForecastingModel:
    """Generate a base model for an individual contributing model.

    Args:
        model_variation (str): Name of the model variation to use.
        contributing_model_kwargs (dict[str, Any]): Keyword arguments to pass to the model constructor.

    Returns:
        GlobalForecastingModel: Base model.
    """
    if model_variation not in MODEL_VARIATIONS:
        raise ValueError(f"Model variation {model_variation} not found.")

    return MODEL_VARIATIONS[model_variation](**contributing_model_kwargs)


def build_model_for_dataset(
    training_dataset: TrainingDataset,
    train_n_points: int,
    contributing_model_variation: str,
    contributing_model_kwargs: dict
) -> RegressionEnsembleModel:
    """Build the EnsembleModel with the contributing models.

    Args:
        training_dataset (TrainingDataset): TrainingDataset instance that will be used to train the models.
        train_n_points (int): Number of points to train regression model on.
        contributing_model_variation (str): Name of the model variation to use.
        contributing_model_kwargs (dict): Keyword arguments to pass to the model constructor.

    Returns:
        RegressionEnsembleModel: Built ensemble model.
    """
    model = RegressionEnsembleModel(
        [
            ContributingModel(generate_base_contributing_model(contributing_model_variation, contributing_model_kwargs), prefix)
            for prefix in training_dataset.subsets
        ],
        train_n_points
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


def run_grid_search_job(parameters: dict[str, Any], working_dir: str, job_id: int):
    """Run a grid search job with the given parameters.

    Args:
        parameters (dict[str, Any]): Parameters to use for the grid search.
        working_dir (str): Working directory to save the trained models to.
        job_id (int): ID of the job. Used to save the model.

    Returns:
        Tuple[float, float]: Score and validation score of the best model.
    """
    model_variation = parameters["model_variation"]
    model_kwargs = {k: v for k, v in parameters.items() if k not in STANDARD_JOB_PARAMETERS}
    coordinates = get_coordinates_for_catchment(parameters["data_file"], parameters["gauge_id"])

    if coordinates is None:
        print(f"Unable to locate {parameters['gauge_id']} in catchment data file.")
        return 1

    columns = get_columns(parameters["columns_file"])

    dataset = get_training_data(parameters["gauge_id"], coordinates, columns)

    model = build_model_for_dataset(dataset, parameters["regression_train_n_points"], model_variation, model_kwargs)

    forecaster = TrainingForecaster(model, dataset, root_dir=f'{working_dir}/trained_models/{str(job_id)}')

    forecaster.fit()
    score = forecaster.backtest_contributing_models()
    val_score = forecaster.backtest_contributing_models(run_on_validation=True)

    return (score, val_score)


def append_scores_to_json(path: str, score: float, val_score: float) -> None:
    """Append the scores to a JSON file.

    Args:
        path (str): Path to the JSON file to append to.
        score (float): Score to append.
        val_score (float): Validation score to append.
    """
    with open(path, 'r') as f:
        data = json.load(f)

    data["score"] = score
    data["val_score"] = val_score

    with open(path, 'w') as f:
        json.dump(data, f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("model_variation", type=str, help="The model variation to use for the grid search.")
    parser.add_argument('-i', '--input_dir', type=str, default='grid_search/jobs', help='Input directory for job JSON files to run')
    parser.add_argument('-j', '--job_id', type=int, default=None, help='Job ID to run. If None, run all jobs in the input directory.')
    args = parser.parse_args()

    working_dir = os.path.join(args.input_dir, args.model_variation)
    job_id = args.job_id

    if job_id is None:
        job_files = os.listdir(working_dir)
        print(job_files)
        for job_file in job_files:
            job_id = job_file.split('.')[0]
            job_filepath = os.path.join(working_dir, job_file)
            with (open(job_filepath)) as f:
                job_data = json.load(f)

            print("Running job: ", job_id)
            print("Using parameters: ", job_data)
            (score, val_score) = run_grid_search_job(job_data, working_dir, job_id)
            append_scores_to_json(job_filepath, score, val_score)
    else:
        job_filepath = os.path.join(working_dir, str(job_id) + '.json')
        with (open(job_filepath)) as f:
            job_data = json.load(f)
        (score, val_score) = run_grid_search_job(job_data, working_dir, job_id)
        append_scores_to_json(job_filepath, score, val_score)
