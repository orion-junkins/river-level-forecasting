#%%
import os
import json
import argparse
import collections

#Parse the command-line argument for the directory path
parser = argparse.ArgumentParser(description='Calculate the average of the contrib test errors field in JSON files.')
parser.add_argument('directory', metavar='dir', type=str, help='the directory containing the JSON files. For example, "grid_search_0_windows/RNN".')
args = parser.parse_args()
base_dir = args.directory


def get_all_scores(jobs_directory):
    """
    Build an ordered dictionary containing all scores found within the jobs directory. Scores will be the keys of the dictionary. The dictionary will be sorted by key.
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
                average_scores[average] =  filename
            except KeyError:
                # If the 'contrib test errors' field doesn't exist, skip the file
                print(f"'{filename}' doesn't have 'contrib test error average' field. Skipping...")
        ordered_average_scores = collections.OrderedDict(sorted(average_scores.items()))
    return ordered_average_scores



def generate_sorted_average_dict(original_dict):
    average_dict = {}
    for key, value in original_dict.items():
        average_value = sum(value) / len(value)
        average_dict[key] = average_value
    sorted_average_dict = dict(sorted(average_dict.items(), key=lambda x: x[1]))
    return sorted_average_dict


# Get all scores for every jobs directory within the base dir and store all scores in a list
rankings = {}
for jobs_dir in os.listdir(base_dir):
    scores = get_all_scores(os.path.join(base_dir, jobs_dir, 'jobs'))
    files_sorted_by_score = list(scores.values())
    for rank, job in enumerate(files_sorted_by_score):
        if job in rankings:
            rankings[job].append(rank)
        else:
            rankings[job] = [rank]

sorted_average_dict = generate_sorted_average_dict(rankings)
#%%

print(sorted_average_dict)