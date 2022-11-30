import pytest

from rlf.forecasting.data_fetching_utilities.coordinate import Coordinate
from rlf.forecasting.data_fetching_utilities.weather_provider.api.base_api_adapter import BaseAPIAdapter
from rlf.forecasting.data_fetching_utilities.weather_provider.api.models import Response
from rlf.forecasting.data_fetching_utilities.weather_provider.api_weather_provider import APIWeatherProvider


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

    def get_index_parameter(self) -> str:
        return "time"


@pytest.fixture
def fake_weather_api_adapter() -> FakeWeatherAPIAdapter:
    return FakeWeatherAPIAdapter()


@ pytest.fixture
def weather_provider(fake_weather_api_adapter) -> APIWeatherProvider:
    coordinates = [Coordinate(lon=1.0, lat=2.0),
                   Coordinate(lon=3.0, lat=4.0)]
    weather_provider = APIWeatherProvider(
        coordinates=coordinates, api_adapter=fake_weather_api_adapter)
    return weather_provider


def test_fetch_historical_datums_fetches_one_per_location(weather_provider):
    weather_datums = weather_provider.fetch_historical_datums()
    assert len(weather_datums) == len(weather_provider.coordinates)


def test_fetch_historical_returns_expected_datum(weather_provider):
    weather_datums = weather_provider.fetch_historical()
    for weather_datum in weather_datums:
        assert weather_datum.hourly_parameters.index.dtype == "datetime64[ns, UTC]"
        assert list(weather_datum.hourly_parameters.columns) == ["temperature_2m"]
        assert len(weather_datum.hourly_parameters) == 3


def test_fetch_current_datums_fetches_one_per_location(weather_provider):
    weather_datums = weather_provider.fetch_current_datums()
    assert len(weather_datums) == len(weather_provider.coordinates)


def test_fetch_current_returns_expected_datum(weather_provider):
    weather_datums = weather_provider.fetch_current()
    for weather_datum in weather_datums:
        assert weather_datum.hourly_parameters.index.dtype == "datetime64[ns, UTC]"
        assert list(weather_datum.hourly_parameters.columns) == ["temperature_2m"]
        assert len(weather_datum.hourly_parameters) == 3
