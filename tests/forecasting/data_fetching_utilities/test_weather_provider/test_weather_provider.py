import pytest

from rlf.forecasting.data_fetching_utilities.coordinate import Coordinate
from rlf.forecasting.data_fetching_utilities.weather_provider.datum import Datum


class FakeWeatherProvider():
    def __init__(self, coordinates: Coordinate):
        self.coordinates = coordinates

    def fetch_historical_weather(self, start_date: str, end_date: str) -> list[Datum]:
        return [Datum(longitude=1.0, latitude=1.0, elevation=3.0,
                      utc_offset_seconds=4.0, timezone="Time Zone", hourly_parameters={})]


@ pytest.fixture
def weather_provider():
    return FakeWeatherProvider(coordinates=[Coordinate(lon=1.0, lat=2.0), Coordinate(lon=3.0, lat=4.0)])


def test_weather_provider_accepts_coordinates_as_list():
    try:
        FakeWeatherProvider(coordinates=[Coordinate(
            lon=1.0, lat=2.0), Coordinate(lon=3.0, lat=4.0)])
    except TypeError:
        pytest.fail("Failed to accept a list of Coordinates")


def test_fetch_historical_weather_returns_list_of_datums(weather_provider):
    datums = weather_provider.fetch_historical_weather(
        start_date="start", end_date="end")
    assert isinstance(datums, list)
    assert isinstance(datums[0], Datum)
