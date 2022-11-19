import pandas as pd
import pytest

from rlf.forecasting.data_fetching_utilities.coordinate import Coordinate
from rlf.forecasting.data_fetching_utilities.weather_provider.api.base_api_adapter import BaseAPIAdapter
from rlf.forecasting.data_fetching_utilities.weather_provider.api.models import Response
from rlf.forecasting.data_fetching_utilities.weather_provider.weather_provider import WeatherProvider
from rlf.aws_dispatcher import AWSDispatcher


def fake_response(coordinate):
    response = Response(status_code=0,
                        url="fake url",
                        message="fake message",
                        headers={"fake": "headers"},
                        data={'latitude': coordinate.lat,
                                'longitude': coordinate.lon,
                                'generationtime_ms': 0.1,
                                'utc_offset_seconds': 0,
                                'timezone': 'GMT',
                                'timezone_abbreviation': 'GMT',
                                'elevation': 123.4,
                                'hourly_units': {'time': 'iso8601', 'temperature_2m': '°C'},
                                'hourly': {'time': ['2000-01-01T00:00',
                                                    '2000-01-01T01:00',
                                                    '2000-01-01T02:00'],
                                            'temperature_2m': [1.0, 2.0, 3.0]}})
    return response


class FakeWeatherAPIAdapter(BaseAPIAdapter):

    def get_current(self, coordinate: Coordinate, **kwargs) -> Response:
        response = fake_response(coordinate=coordinate)
        return response

    def get_historical(self, coordinate: Coordinate, **kwargs) -> Response:
        response = fake_response(coordinate=coordinate)
        return response


@pytest.fixture
def fake_weather_api_adapter() -> FakeWeatherAPIAdapter:
    return FakeWeatherAPIAdapter()


@ pytest.fixture
def weather_provider(fake_weather_api_adapter) -> WeatherProvider:
    coordinates = [Coordinate(lon=1.0, lat=2.0),
                   Coordinate(lon=3.0, lat=4.0)]
    weather_provider = WeatherProvider(
        coordinates=coordinates, api_adapter=fake_weather_api_adapter)
    return weather_provider


def test_fetch_historical_datums_fetches_one_per_location(weather_provider):
    weather_datums = weather_provider.fetch_historical_datums()
    assert len(weather_datums) == len(weather_provider.coordinates)


def test_fetch_historical_returns_expected_df(weather_provider):
    weather_dfs = weather_provider.fetch_historical()
    for weather_df in weather_dfs:
        assert isinstance(weather_df, pd.DataFrame)
        assert list(weather_df.columns) == ["time", "temperature_2m"]
        assert len(weather_df) == 3


def test_fetch_current_datums_fetches_one_per_location(weather_provider):
    weather_datums = weather_provider.fetch_current_datums()
    assert len(weather_datums) == len(weather_provider.coordinates)


def test_fetch_current_returns_expected_df(weather_provider):
    weather_dfs = weather_provider.fetch_current()
    for weather_df in weather_dfs:
        assert isinstance(weather_df, pd.DataFrame)
        assert list(weather_df.columns) == ["time", "temperature_2m"]
        assert len(weather_df) == 3


@pytest.fixture
def aws_dispatcher():
    return AWSDispatcher(bucket_name="testing-bucket-junkinso", directory_name="weather_provider_testing")


@ pytest.fixture
def aws_weather_provider(fake_weather_api_adapter, aws_dispatcher) -> WeatherProvider:
    coordinates = [Coordinate(lon=1.0, lat=2.0),
                   Coordinate(lon=3.0, lat=4.0)]
    weather_provider = WeatherProvider(
        coordinates=coordinates, api_adapter=fake_weather_api_adapter, aws_dispatcher=aws_dispatcher)
    return weather_provider


@pytest.mark.slow
def test_historical_data_aws(aws_weather_provider):
    aws_weather_provider.update_historical_datums_in_aws()
    datums = aws_weather_provider.download_historical_datums_from_aws()
    for datum in datums:
        assert isinstance(datum.hourly_parameters, pd.DataFrame)
        assert list(datum.hourly_parameters.columns) == ["time", "temperature_2m"]
        assert len(datum.hourly_parameters) == 3


@pytest.mark.slow
def test_fetch_historical_aws(aws_weather_provider):
    weather_dfs = aws_weather_provider.fetch_historical()
    for weather_df in weather_dfs:
        assert isinstance(weather_df, pd.DataFrame)
        assert list(weather_df.columns) == ["time", "temperature_2m"]
        assert len(weather_df) == 3
