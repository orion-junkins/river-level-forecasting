import argparse
import json
import os
import time
from typing import Any
import itertools

try:
    from darts.models.forecasting.rnn_model import RNNModel
except ImportError as e:
    print("Import error on darts packages. Ensure darts and its dependencies have been installed into the local environment.")
    print(e)
    exit(1)


# Mapping of model variation names to their corresponding Darts classes.
MODEL_VARIATIONS = {
    "RNN": RNNModel
}


def build_json_jobs(search_space: dict[str, Any], dir: str = "gs_jobs"):
    """
    Build a JSON file containing all jobs from a search space.

    Args:
        search_space (dict[str, Any]): The search space to explore.
        dir (str, optional): Directory to save the JSON file in. Defaults to "gs_jobs".
    """
    os.makedirs(dir, exist_ok=True)
    keys, values = zip(*search_space.items())
    permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]

    for i, permutation in enumerate(permutations_dicts):
        permutation['model_variation'] = MODEL_VARIATION
        cur_path = os.path.join(dir, f"{i}.json")
        with open(cur_path, "w") as f:
            json.dump(permutation, f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("model_variation")
    parser.add_argument('-s', '--search_space', type=str, default='data/grid_search_space.json', help='JSON file containing all grid search spaces')
    parser.add_argument('-o', '--output_dir', type=str, default='grid_search/jobs', help='=Ouput directory for generated job JSON files')
    args = parser.parse_args()

    MODEL_VARIATION = args.model_variation
    SEARCH_SPACE_FILEPATH = args.search_space
    OUTPUT_DIR = os.path.join(args.output_dir, MODEL_VARIATION)
    current_timestamp = str(time.time())

    with open(SEARCH_SPACE_FILEPATH) as json_file:
        all_search_spaces = json.load(json_file)
    cur_search_space = all_search_spaces[MODEL_VARIATION]
    build_json_jobs(cur_search_space, OUTPUT_DIR)
