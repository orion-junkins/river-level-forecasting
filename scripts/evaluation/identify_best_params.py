import os
import json
import argparse
import collections

# Parse the command-line argument for the directory path
parser = argparse.ArgumentParser(description='Calculate the average of the contrib test errors field in JSON files.')
parser.add_argument('directory', metavar='dir', type=str, help='the directory containing the JSON files. For example, "grid_search_0_windows/RNN".')
args = parser.parse_args()
base_dir = args.directory


def get_all_scores(jobs_directory: str) -> dict[float, str]:
    """
    Build an ordered dictionary containing all scores found within a single jobs directory. Scores will be the keys of the dictionary. The dictionary will be sorted by key. This is useful for finding the best scoring jobs in the directory.
    """
    # Create an empty dictionary to store the averages
    average_scores = {}

    # Iterate through the files in the jobs_directory
    for filename in os.listdir(jobs_directory):
        # Check if the file is a JSON file
        if filename.endswith('.json'):
            # Load the JSON file
            with open(os.path.join(jobs_directory, filename), 'r') as f:
                data = json.load(f)
            # Grab the average contrib test error
            try:
                average = data["contrib test error average"]
                # Add the average to the dictionary
                average_scores[average] = filename
            except KeyError:
                # If the 'contrib test errors' field doesn't exist, skip the file
                print(f"'{filename}' doesn't have 'contrib test error average' field. Skipping...")
        ordered_average_scores = collections.OrderedDict(sorted(average_scores.items()))
    return ordered_average_scores


def generate_sorted_average_ranking(original_dict: dict[str, list[int]]) -> dict[str, float]:
    """
    Generate a dictionary where the keys are the job ids and the values are the average scores.

    Args:
        original_dict (dict[str, list[float]]): A dictionary where the keys are the job ids and the values are lists of scores.

    Returns:
        dict[str, float]: A dictionary where the keys are the job ids and the values are the average scores.
    """
    average_dict = {}
    for key, value in original_dict.items():
        average_value = sum(value) / len(value)
        average_dict[key] = average_value
    sorted_average_dict = dict(sorted(average_dict.items(), key=lambda x: x[1]))
    return sorted_average_dict


# Get all scores for every jobs directory within the base dir
# Store all scores for each job variation across all jobs directory in a list
# Build a rankings dictionary where the keys are the job filenames and the values are lists of the rankings achieved by that job across all jobs directories
rankings: dict[str, list[int]] = {}
for jobs_dir in os.listdir(base_dir):
    scores = get_all_scores(os.path.join(base_dir, jobs_dir, 'jobs'))
    files_sorted_by_score = list(scores.values())
    for rank, job in enumerate(files_sorted_by_score):
        if job in rankings:
            rankings[job].append(rank)
        else:
            rankings[job] = [rank]


# Average and sort the rankings
sorted_average_ranking = generate_sorted_average_ranking(rankings)


# Print the sorted average ranking
print(sorted_average_ranking)
