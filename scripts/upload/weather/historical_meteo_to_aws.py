# Helper script for uploading historical weather from OpenMeteo to AWS. Intended to be modified & run as needed to upload new historical datasets.
# %%
from rlf.aws_dispatcher import AWSDispatcher
from rlf.forecasting.data_fetching_utilities.coordinate import Coordinate
from rlf.forecasting.data_fetching_utilities.weather_provider.api_weather_provider import APIWeatherProvider
from rlf.forecasting.data_fetching_utilities.weather_provider.aws_weather_uploader import AWSWeatherUploader

# Tunable parameters
coordinates = [Coordinate(lon=-120.75, lat=44.25), Coordinate(lon=-121.0, lat=44.5)]
start_date = "2010-01-01"
end_date = "2021-01-01"
sleep_duration = 5
bucket_name = "all-weather-data"
directory_name = "open-meteo"

api_weather_provider = APIWeatherProvider(coordinates=coordinates)

aws_dispatcher = AWSDispatcher(bucket_name=bucket_name, directory_name=directory_name)

aws_weather_uploader = AWSWeatherUploader(weather_provider=api_weather_provider, aws_dispatcher=aws_dispatcher)

aws_weather_uploader.upload_historical(start_date=start_date, end_date=end_date, sleep_duration=sleep_duration)

# %%
