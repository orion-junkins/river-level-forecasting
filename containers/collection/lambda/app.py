from datetime import datetime

from rlf.aws_dispatcher import AWSDispatcher
from rlf.forecasting.data_fetching_utilities.coordinate import Coordinate
from rlf.forecasting.data_fetching_utilities.weather_provider.api_weather_provider import APIWeatherProvider
from rlf.forecasting.data_fetching_utilities.weather_provider.aws_weather_uploader import AWSWeatherUploader
ALL_COORDINATES = list(
    map(lambda x: Coordinate(x[0], x[1]),
        [
        [
            -122.4,
            45.8
        ],
        [
            -122.3,
            45.8
        ],
        [
            -122.2,
            45.8
        ]
    ]
    ))


def handler(event, context):
    print(event)
    now = datetime.utcnow()
    timestamp = now.strftime("%y-%m-%d_%H-%M")
    current_timestamp = str(timestamp)

    MIN_IDX = 0
    MAX_IDX = 10
    TIMESTAMP = current_timestamp

    # Tunable parameters
    SLEEP_DURATION = 0.0
    BUCKET_NAME = "all-weather-data"
    AWS_DIR_NAME = "open-meteo"

    coordinates = ALL_COORDINATES[MIN_IDX:MAX_IDX]
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
