# Helper script for uploading historical weather from OpenMeteo to AWS. Intended to be modified & run as needed to upload new historical datasets.
import json
import sys

from rlf.aws_dispatcher import AWSDispatcher
from rlf.forecasting.data_fetching_utilities.coordinate import Coordinate
from rlf.forecasting.data_fetching_utilities.weather_provider.api_weather_provider_ecmwf import APIWeatherProviderECMWF
from rlf.forecasting.data_fetching_utilities.weather_provider.aws_weather_uploader import AWSWeatherUploader

# Parse command line args
args = [arg for arg in sys.argv[1:] if not arg.startswith("-")]

if len(args) == 3:
    CATCHMENT_FILEPATH = args[0]
    START_DATE = args[1]
    END_DATE = args[2]
else:
    raise ValueError(f'Usage: {sys.argv[0]} CATCHMENT_FILEPATH START_DATE END_DATE, where dates are given in the form "yyyy-mm-dd"')

# Tunable parameters
SLEEP_DURATION = 21
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

print(f'Uploading historical weather data for {len(coordinates)} points')

# Instantiate an APIWeatherProvider, AWSDispatcher, and AWSWeatherUploader
api_weather_provider = APIWeatherProviderECMWF(coordinates=coordinates)
aws_dispatcher = AWSDispatcher(bucket_name=BUCKET_NAME, directory_name=AWS_DIR_NAME)
aws_weather_uploader = AWSWeatherUploader(weather_provider=api_weather_provider, aws_dispatcher=aws_dispatcher)

# Upload historical weather to AWS
aws_weather_uploader.upload_historical(start_date=START_DATE, end_date=END_DATE, sleep_duration=SLEEP_DURATION)
