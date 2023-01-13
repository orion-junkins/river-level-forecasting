# Helper script for uploading current weather from OpenMeteo to AWS. Intended to be run as scheduled lambda function to log evaluation datasets, or locally for testing purposes.
# %%
from datetime import datetime

from rlf.aws_dispatcher import AWSDispatcher
from rlf.forecasting.data_fetching_utilities.coordinate import Coordinate
from rlf.forecasting.data_fetching_utilities.weather_provider.api_weather_provider import APIWeatherProvider
from rlf.forecasting.data_fetching_utilities.weather_provider.aws_weather_uploader import AWSWeatherUploader

# Tunable parameters
coordinates = [Coordinate(lon=-120.75, lat=44.25), Coordinate(lon=-121.0, lat=44.5)]
sleep_duration = 5
bucket_name = "all-weather-data"
directory_name = "open-meteo"

# Generate timestamp string for directory naming
now = datetime.now()
timestamp = now.strftime("%y-%m-%d_%H-%M")
dir_path = str(timestamp)

api_weather_provider = APIWeatherProvider(coordinates=coordinates)

aws_dispatcher = AWSDispatcher(bucket_name=bucket_name, directory_name=directory_name)

aws_weather_uploader = AWSWeatherUploader(weather_provider=api_weather_provider, aws_dispatcher=aws_dispatcher)

aws_weather_uploader.upload_current(dir_path=dir_path)

# %%
