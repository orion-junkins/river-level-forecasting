# Helper script for uploading current weather from OpenMeteo to AWS. Intended to be run as scheduled lambda function to log evaluation datasets, or locally for testing purposes.
from datetime import datetime
import json
import sys

from rlf.aws_dispatcher import AWSDispatcher
from rlf.forecasting.data_fetching_utilities.coordinate import Coordinate
from rlf.forecasting.data_fetching_utilities.weather_provider.api_weather_provider_ecmwf import APIWeatherProviderECMWF
from rlf.forecasting.data_fetching_utilities.weather_provider.aws_weather_uploader import AWSWeatherUploader

# Parse command line args
args = [arg for arg in sys.argv[1:] if not arg.startswith("-")]

# Use provided path if one is given, else default to expected path in data dir
if len(args) == 1:
    CATCHMENT_FILEPATH = args[0]
else:
    CATCHMENT_FILEPATH = "data/catchments_test.json"

# Tunable parameters
SLEEP_DURATION = 0.2
BUCKET_NAME = "ecmwf-weather-data"
AWS_DIR_NAME = "open-meteo"

# Load a list of Coordinate objects from the json file at the given path
with open(CATCHMENT_FILEPATH) as f:
    data = json.load(f)
coordinates_raw = []
for feature in data["features"]:
    coordinates_raw.extend(feature["geometry"]["coordinates"])
coordinates = []
for coord in coordinates_raw:
    new_coord = Coordinate(lon=coord[0], lat=coord[1])
    coordinates.append(new_coord)
coordinates = list(set(coordinates))

print(f'Uploading current weather data for {len(coordinates)} points')

# Generate timestamp string for directory naming
now = datetime.now()
timestamp = now.strftime("%y-%m-%d_%H-%M")
dir_path = str(timestamp)

# Instantiate an APIWeatherProvider, AWSDispatcher, and AWSWeatherUploader
api_weather_provider = APIWeatherProviderECMWF(coordinates=coordinates)
aws_dispatcher = AWSDispatcher(bucket_name=BUCKET_NAME, directory_name=AWS_DIR_NAME)
aws_weather_uploader = AWSWeatherUploader(weather_provider=api_weather_provider, aws_dispatcher=aws_dispatcher)

# Upload current weather to AWS
aws_weather_uploader.upload_current(dir_path=dir_path, sleep_duration=SLEEP_DURATION)
