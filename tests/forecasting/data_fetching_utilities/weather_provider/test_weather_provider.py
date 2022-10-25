import pytest

from rlf.forecasting.data_fetching_utilities.coordinate import Coordinate
from rlf.forecasting.data_fetching_utilities.weather_provider.api.base_api_adapter import BaseAPIAdapter
from rlf.forecasting.data_fetching_utilities.weather_provider.weather_datum import WeatherDatum
from rlf.forecasting.data_fetching_utilities.weather_provider.api.models import Response
from rlf.forecasting.data_fetching_utilities.weather_provider.weather_provider import WeatherProvider


class FakeWeatherAPIAdapter(BaseAPIAdapter):

    def __init__(self,
                 longitude: float,
                 latitude: float,
                 start_date: str,
                 end_date: str,
                 protocol: str = "fake protocol",
                 hostname: str = "fake hostname",
                 version: str = "fake version",
                 path: str = "fake path",
                 hourly_parameters: list[str] = ["fake weather variable"]) -> None:
        self.longitude = longitude
        self.latitude = latitude
        self.start_date = start_date
        self.end_date = end_date
        self.protocol = protocol
        self.hostname = hostname
        self.version = version
        self.path = path
        self.hourly_parameters = hourly_parameters

    def get(self) -> Response:
        response = Response(status_code=0,
                            url="fake url",
                            message="fake message",
                            headers={"fake": "headers"},
                            data={"longitude": self.longitude,
                                  "latitude": self.latitude,
                                  "hourly": self.hourly_parameters})
        return response


@pytest.fixture
def fake_weather_api_adapter() -> FakeWeatherAPIAdapter:
    return FakeWeatherAPIAdapter(longitude=0.0, latitude=0.0, start_date="fake start", end_date="fake end")


@ pytest.fixture
def weather_provider(fake_weather_api_adapter) -> WeatherProvider:
    coordinates = [Coordinate(lon=1.0, lat=2.0),
                   Coordinate(lon=3.0, lat=4.0)]
    weather_provider = WeatherProvider(
        coordinates=coordinates, api_adapter=fake_weather_api_adapter)
    return weather_provider


def test_fetch_historical_input_output_equal(weather_provider):
    weather_datums = weather_provider.fetch_historical_weather(
        start_date="fake start", end_date="fake end")
    assert len(weather_datums) == 2


def test_fetch_historical_returns_weather_datum(weather_provider):
    weather_datums = weather_provider.fetch_historical_weather(
        start_date="fake start", end_date="fake end")
    assert isinstance(weather_datums[0], WeatherDatum)


def test_fetch_historical_returns_datum_at_input_location(weather_provider):
    weather_datums = weather_provider.fetch_historical_weather(
        start_date="fake start", end_date="fake end")
    assert weather_datums[0].longitude == 1.0
    assert weather_datums[0].latitude == 2.0
    assert weather_datums[0].hourly_parameters == ["fake weather variable"]

    assert weather_datums[1].longitude == 3.0
    assert weather_datums[1].latitude == 4.0
    assert weather_datums[1].hourly_parameters == ["fake weather variable"]
