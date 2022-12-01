import pytest

from rlf.aws_dispatcher import AWSDispatcher
from rlf.forecasting.data_fetching_utilities.coordinate import Coordinate
from rlf.forecasting.data_fetching_utilities.weather_provider.aws_weather_provider import AWSWeatherProvider


@pytest.fixture
def aws_dispatcher() -> AWSDispatcher:
    return AWSDispatcher("historical-weather", "testing")


@ pytest.fixture
def weather_provider(aws_dispatcher) -> AWSWeatherProvider:
    coordinates = [Coordinate(lon=1.0, lat=2.0),
                   Coordinate(lon=3.0, lat=4.0)]
    weather_provider = AWSWeatherProvider(
        coordinates=coordinates, aws_dispatcher=aws_dispatcher)
    return weather_provider


def test_fetch_historical(weather_provider):
    weather_datums = weather_provider.fetch_historical()
    assert len(weather_datums) == len(weather_provider.coordinates)
    for weather_datum in weather_datums:
        assert weather_datum.hourly_parameters.index.dtype == "datetime64[ns, UTC]"
        assert ("temperature_2m" in list(weather_datum.hourly_parameters.columns))
