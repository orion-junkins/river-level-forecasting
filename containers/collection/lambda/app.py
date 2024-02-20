import json
from datetime import datetime

from rlf.aws_dispatcher import AWSDispatcher
from rlf.forecasting.data_fetching_utilities.coordinate import Coordinate
from rlf.forecasting.data_fetching_utilities.weather_provider.api_weather_provider import APIWeatherProvider
from rlf.forecasting.data_fetching_utilities.weather_provider.aws_weather_uploader import AWSWeatherUploader


def handler(event, context):
    CATCHMENT_FILEPATH = "data/catchments_short.json"
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

    now = datetime.utcnow()
    timestamp = now.strftime("%y-%m-%d_%H-%M")
    current_timestamp = str(timestamp)

    TIMESTAMP = current_timestamp

    # Tunable parameters
    SLEEP_DURATION = 0.0
    BUCKET_NAME = "all-weather-data"
    AWS_DIR_NAME = "open-meteo"

    print(f'Uploading current weather data for {len(coordinates)} points')

    # Generate timestamp string for directory naming
    dir_path = str(TIMESTAMP)

    # Instantiate an APIWeatherProvider, AWSDispatcher, and AWSWeatherUploader
    api_weather_provider = APIWeatherProvider(coordinates=coordinates)
    aws_dispatcher = AWSDispatcher(bucket_name=BUCKET_NAME, directory_name=AWS_DIR_NAME)
    aws_weather_uploader = AWSWeatherUploader(weather_provider=api_weather_provider, aws_dispatcher=aws_dispatcher)

    # Upload current weather to AWS
    aws_weather_uploader.upload_current(dir_path=dir_path, sleep_duration=SLEEP_DURATION)

    return


if __name__ == "__main__":
    handler(None, None)
