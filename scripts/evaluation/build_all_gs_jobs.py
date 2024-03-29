import argparse
import json
import os
import time
from typing import Any, Dict
import itertools
import sys

try:
    from darts.models.forecasting.rnn_model import RNNModel
    from darts.models.forecasting.transformer_model import TransformerModel
    from darts.models.forecasting.nhits import NHiTSModel
    from darts.models.forecasting.nbeats import NBEATSModel
except ImportError as e:
    print("Import error on darts packages. Ensure darts and its dependencies have been installed into the local environment.")
    print(e)
    exit(1)


# Mapping of model variation names to their corresponding Darts classes.
MODEL_VARIATIONS = {
    "RNN": RNNModel,
    "Transformer": TransformerModel,
    "NHiTS": NHiTSModel,
    "NBEATS": NBEATSModel
}


def build_json_jobs(search_space: Dict[str, Any], dir: str = "grid_search", purge_directory: bool = False) -> None:
    """
    Build a JSON file containing all jobs from a search space.

    Args:
        search_space (Dict[str, Any]): The search space to explore.
        dir (str, optional): Directory to save the JSON file in. Defaults to "grid_search".
        purge_directory (bool, optional): If True then purge the directory before generating jobs. Defaults to False.
    """
    model_variation = search_space['contributing_model_type']
    gauge_id = search_space['gauge_id'][0]

    dir = os.path.join(dir, model_variation, gauge_id, "jobs")
    os.makedirs(dir, exist_ok=True)
    if purge_directory:
        for file in os.listdir(dir):
            os.remove(os.path.join(dir, file))

    keys, values = zip(*search_space.items())
    permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]

    for i, permutation in enumerate(permutations_dicts):
        permutation['contributing_model_type'] = model_variation
        cur_path = os.path.join(dir, f"{i}.json")
        with open(cur_path, "w") as f:
            json.dump(permutation, f, indent=4)


if __name__ == '__main__':
    # Parse the command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("model_variation", type=str, help="The model variation to use for the grid search.")
    parser.add_argument("gauge_id", type=str, help="The ID of the USGS gauge to use for the grid search.")
    parser.add_argument('-s', '--search_space', type=str, default='data/grid_search_0_model.json', help='JSON file containing all grid search spaces')
    parser.add_argument('-o', '--output_dir', type=str, default='grid_search', help='Ouput directory for generated job JSON files')
    parser.add_argument('-p', '--purge_output_dir', type=bool, default=False, help='=Purge the output directory before generating jobs')

    args = parser.parse_args()

    model_variation = args.model_variation
    gauge_id = args.gauge_id
    search_space_filepath = args.search_space
    output_dir = args.output_dir
    purge_directory = bool(args.purge_output_dir)

    current_timestamp = str(time.time())

    # Load the search space for the specified model variation
    with open(search_space_filepath) as json_file:
        all_search_spaces = json.load(json_file)
    search_space = all_search_spaces[model_variation]

    # Set up the output directory
    dir = os.path.join(output_dir, model_variation, gauge_id, "jobs")
    os.makedirs(dir, exist_ok=True)
    if purge_directory:
        for file in os.listdir(dir):
            os.remove(os.path.join(dir, file))
    elif len(os.listdir(dir)) > 0:
        print("Output directory is not empty. Please purge it before generating jobs.")
        sys.exit(1)

    # Generate the job permutations
    keys, values = zip(*search_space.items())
    permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]

    # Save the job permutations to JSON files
    for i, permutation in enumerate(permutations_dicts):
        permutation['contributing_model_type'] = model_variation
        permutation['gauge_id'] = gauge_id
        cur_path = os.path.join(dir, f"{i}.json")
        with open(cur_path, "w") as f:
            json.dump(permutation, f, indent=4)
