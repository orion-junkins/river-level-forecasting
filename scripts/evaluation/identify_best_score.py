import os
import json
import argparse

# Parse the command-line argument for the directory path
parser = argparse.ArgumentParser(description='Calculate the average of the contrib test errors field in JSON files.')
parser.add_argument('directory', metavar='dir', type=str, help='the directory containing the JSON files')
args = parser.parse_args()

# Create an empty dictionary to store the average score for each file
file_averages: dict[str, float] = {}

# Iterate through the files in the directory
for filename in os.listdir(args.directory):
    # Check if the file is a JSON file
    if filename.endswith('.json'):
        # Load the JSON file
        with open(os.path.join(args.directory, filename), 'r') as f:
            data = json.load(f)
        # Grab the 'contrib test error average' field
        try:
            average = data["contrib test error average"]
            # Add the average to the dictionary
            file_averages[filename] = average
        except KeyError:
            # If the 'contrib test errors' field doesn't exist, skip the file
            print(f"'{filename}' doesn't have 'contrib test error average' field. Skipping...")

# Print the names of the 3 files with the lowest average
if file_averages:
    sorted_files = sorted(file_averages.items(), key=lambda x: x[1])
    for i in range(min(3, len(sorted_files))):
        print(sorted_files[i][0], "---", sorted_files[i][1])
else:
    print("No files with 'contrib test errors' field found.")
