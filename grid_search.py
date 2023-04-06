import argparse
import json
import os
from multiprocessing import Pool
import time
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
STANDARD_JOB_PARAMETERS = ["data_file", "columns_file", "gauge_id", "regression_train_n_points"]

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


def run_grid_search_job(model_variation: str, parameters: dict[str, Any], job_id: int):
    """Run a grid search job with the given parameters.

    Args:
        model_variation (str): Name of the model variation to use.
        parameters (dict[str, Any]): Parameters to use for the grid search.
        job_id (int): ID of the job. Used to save the model.

    Returns:
        Tuple[float, float]: Score and validation score of the best model.
    """
    model_kwargs = {k: v for k, v in parameters.items() if k not in STANDARD_JOB_PARAMETERS}
    coordinates = get_coordinates_for_catchment(parameters["data_file"], parameters["gauge_id"])

    if coordinates is None:
        print(f"Unable to locate {parameters['gauge_id']} in catchment data file.")
        return 1
    columns = get_columns(parameters["columns_file"])
    dataset = get_training_data(parameters["gauge_id"], coordinates, columns)
    model = build_model_for_dataset(dataset, parameters["regression_train_n_points"], model_variation, model_kwargs)
    forecaster = TrainingForecaster(model, dataset, root_dir=f'trained_models/grid_search_models/{model_variation}/{str(job_id)}')
    forecaster.fit()
    score = forecaster.backtest()
    val_score = forecaster.backtest(run_on_validation=True)

    return (score, val_score)


def dispatch_job(job_data: tuple[dict[str, Any], int, str, str]):
    """
    Dispatch a single grid search job with the given set of data.

    Args:
        job_data (tuple[dict[str, Any], int, str, str]): All data needed for the job. Expected to be a tuple of parameters, job ID, model variation, and output directory.

    Raises:
        ValueError: Raised if any of the parameters have more than one value.

    Returns:
        _type_: _description_
    """
    (raw_job_parameters, id, model_variation, out_dir) = job_data
    print("Spawning job " + str(id) + " with parameters: " + str(raw_job_parameters))

    job_parameters = {}
    for key, value in raw_job_parameters.items():
        if len(value) > 1:
            raise ValueError("Search space must be a single element list for all values.")
        job_parameters[key] = value[0]
    output_dict = {"parameters": job_parameters}

    out_dir = os.path.join(out_dir, job_parameters["gauge_id"])
    os.makedirs(out_dir, exist_ok=True)
    output_path = os.path.join(out_dir, f"{id}.json")

    (score, val_score) = run_grid_search_job(model_variation, job_parameters, id)
    output_dict["score"] = score
    output_dict["val_score"] = val_score

    with open(output_path, "w") as f:
        json.dump(output_dict, f, indent=4)

    return 0


def recursive_job_builder(search_space: dict[str, Any]) -> list[dict[str, List[Any]]]:
    """
    Recursively build a list of jobs from a search space. All values in the search space must be lists. If any value is a list with more than one element, the first element is removed and a job is dispatched for each remaining element. This is repeated until all values are single element lists.

    Args:
        search_space (dict[str, Any]): The search space to explore.

    Returns:
        list[dict[str, List[Any]]]: List of jobs to dispatch.
    """
    all_jobs = []

    # Check base case (all single element lists)
    multi_element_lists_remain = True in (len(value) > 1 for value in search_space.values())
    if not multi_element_lists_remain:
        all_jobs = [search_space]
    else:
        # Otherwise, split the first multi-element list and dispatch jobs for each
        for key, value in search_space.items():
            if len(value) > 1:
                new_search_space_first_elem = search_space.copy()
                new_search_space_first_elem[key] = value[1:]
                all_jobs.extend(recursive_job_builder(new_search_space_first_elem))

                new_search_space_other_elems = search_space.copy()
                new_search_space_other_elems[key] = value[:1]
                all_jobs.extend(recursive_job_builder(new_search_space_other_elems))
    return all_jobs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("model_variation")
    parser.add_argument('-j', '--num_jobs', type=int, default=3, help='the number of gridsearch jobs to run in parallel')
    parser.add_argument('-s', '--search_space', type=str, default='data/grid_search_space_test.json', help='JSON file containing all grid search spaces')
    args = parser.parse_args()

    MODEL_VARIATION = args.model_variation
    SEARCH_SPACE_FILEPATH = args.search_space
    current_timestamp = str(time.time())
    OUT_DIR = os.path.join("GS_OUTPUTS", current_timestamp, MODEL_VARIATION)
    NUM_JOBS = args.num_jobs

    with open(SEARCH_SPACE_FILEPATH) as json_file:
        all_search_spaces = json.load(json_file)
    cur_search_space = all_search_spaces[MODEL_VARIATION]

    jobs = recursive_job_builder(cur_search_space)
    print("Dispatching " + str(len(jobs)) + " jobs. Running up to " + str(NUM_JOBS) + " in parallel.")

    with Pool(NUM_JOBS) as p:
        result = p.map(dispatch_job, [(job, id, MODEL_VARIATION, OUT_DIR) for id, job in enumerate(jobs)])
