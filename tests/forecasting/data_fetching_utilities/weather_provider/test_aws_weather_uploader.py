from datetime import datetime, timedelta
import pandas as pd
import pytest

from rlf.aws_dispatcher import AWSDispatcher
from rlf.forecasting.data_fetching_utilities.coordinate import Coordinate
from rlf.forecasting.data_fetching_utilities.weather_provider.aws_weather_provider import AWSWeatherProvider
from rlf.forecasting.data_fetching_utilities.weather_provider.api_weather_provider import APIWeatherProvider
from rlf.forecasting.data_fetching_utilities.weather_provider.aws_weather_uploader import AWSWeatherUploader

DEFAULT_START_DATE = "2020-01-01"
END_DATE = "2020-02-01"
CURRENT_TIMESTAMP = "22-12-04_09-25"


@pytest.fixture
def coordinates() -> list[Coordinate]:
    return [Coordinate(lon=-120.8, lat=44.2), Coordinate(lon=-121.8, lat=44.3)]


@pytest.fixture
def api_weather_provider(coordinates):
    return APIWeatherProvider(coordinates=coordinates)


@pytest.fixture
def aws_dispatcher() -> AWSDispatcher:
    return AWSDispatcher("all-weather-data", "testing")


@pytest.fixture
def aws_weather_uploader(api_weather_provider, aws_dispatcher):
    return AWSWeatherUploader(weather_provider=api_weather_provider, aws_dispatcher=aws_dispatcher)


@ pytest.fixture
def weather_provider(coordinates, aws_dispatcher) -> AWSWeatherProvider:
    weather_provider = AWSWeatherProvider(
        coordinates=coordinates, aws_dispatcher=aws_dispatcher)
    return weather_provider


@pytest.mark.slow
def test_upload_historical(aws_weather_uploader, weather_provider):
    aws_weather_uploader.upload_historical(start_date=DEFAULT_START_DATE, end_date=END_DATE)
    weather_datums = weather_provider.fetch_historical()
    assert len(weather_datums) == len(weather_provider.coordinates)

    expected_start_date = pd.to_datetime(DEFAULT_START_DATE, format='%Y-%m-%d')
    expected_end_date = pd.to_datetime(END_DATE, format='%Y-%m-%d')

    for weather_datum in weather_datums:
        actual_start_date = weather_datum.hourly_parameters.index[0]
        actual_end_date = weather_datum.hourly_parameters.index[-1]

        assert (expected_start_date.year == actual_start_date.year)
        assert (expected_start_date.month == actual_start_date.month)
        assert (expected_start_date.day == actual_start_date.day)

        assert (expected_end_date.year == actual_end_date.year)
        assert (expected_end_date.month == actual_end_date.month)
        assert (expected_end_date.day == actual_end_date.day)


@pytest.mark.slow
def test_upload_current(aws_weather_uploader, weather_provider):
    # Generate timestamp string for directory naming
    now = datetime.now()
    timestamp = now.strftime("%y-%m-%d_%H-%M")
    dir_path = str(timestamp)

    expected_start_date = now - timedelta(days=92)
    expected_end_date = now + timedelta(days=15)

    aws_weather_uploader.upload_current(dir_path=dir_path)
    weather_provider.set_timestamp(dir_path)
    weather_datums = weather_provider.fetch_current()
    assert len(weather_datums) == len(weather_provider.coordinates)

    for weather_datum in weather_datums:
        actual_start_date = weather_datum.hourly_parameters.index[0]
        actual_end_date = weather_datum.hourly_parameters.index[-1]

        assert (expected_start_date.year == actual_start_date.year)
        assert (expected_start_date.month == actual_start_date.month)
        assert (expected_start_date.day == actual_start_date.day)

        assert (expected_end_date.year == actual_end_date.year)
        assert (expected_end_date.month == actual_end_date.month)
        assert (expected_end_date.day == actual_end_date.day)
