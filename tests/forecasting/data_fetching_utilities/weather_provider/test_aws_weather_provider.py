import pytest

from rlf.aws_dispatcher import AWSDispatcher
from rlf.forecasting.data_fetching_utilities.coordinate import Coordinate
from rlf.forecasting.data_fetching_utilities.weather_provider.aws_weather_provider import AWSWeatherProvider


@pytest.fixture
def aws_dispatcher() -> AWSDispatcher:
    return AWSDispatcher("all-weather-data", "testing")


@pytest.fixture
def coordinates() -> list[Coordinate]:
    return [Coordinate(lon=-120.75, lat=44.25), Coordinate(lon=-121.0, lat=44.5)]


@ pytest.fixture
def weather_provider(aws_dispatcher, coordinates) -> AWSWeatherProvider:
    return AWSWeatherProvider(
        coordinates=coordinates, aws_dispatcher=aws_dispatcher)


@pytest.mark.slow
def test_fetch_historical(weather_provider):
    weather_datums = weather_provider.fetch_historical()
    assert len(weather_datums) == len(weather_provider.coordinates)
    for weather_datum in weather_datums:
        assert weather_datum.hourly_parameters.index.dtype == "datetime64[ns, UTC]"
        assert ("temperature_2m" in list(weather_datum.hourly_parameters.columns))


@pytest.mark.slow
def test_fetch_current(weather_provider):
    weather_provider.set_timestamp("22-12-07_15-14")
    weather_datums = weather_provider.fetch_current()
    assert len(weather_datums) == len(weather_provider.coordinates)
    for weather_datum in weather_datums:
        assert weather_datum.hourly_parameters.index.dtype == "datetime64[ns, UTC]"
        assert ("temperature_2m" in list(weather_datum.hourly_parameters.columns))
