import pytest

from rlf.forecasting.data_fetching_utilities.coordinate import Coordinate
from rlf.forecasting.data_fetching_utilities.weather_provider.api_weather_provider_ecmwf import APIWeatherProviderECMWF


@ pytest.fixture
def weather_provider_ecmwf() -> APIWeatherProviderECMWF:
    coordinates = [Coordinate(lon=-121.6, lat=47.3),
                   Coordinate(lon=-121.3, lat=47.4)]
    weather_provider = APIWeatherProviderECMWF(
        coordinates=coordinates)
    return weather_provider


def test_initialization(weather_provider_ecmwf):

    assert weather_provider_ecmwf is not None


def test_fetch_current(weather_provider_ecmwf):
    test_list = weather_provider_ecmwf.fetch_current(["temperature_2m", "rain"])

    assert test_list is not None
    assert len(test_list) == 2
    assert test_list[0].hourly_parameters is not None
    assert list(test_list[0].hourly_parameters.columns) == ["temperature_2m", "rain"]


def test_fetch_current_equivalence(weather_provider_ecmwf):
    test_list_1 = weather_provider_ecmwf.fetch_current(["temperature_2m", "rain"])
    test_list_2 = weather_provider_ecmwf.fetch_current(["temperature_2m", "rain"])

    assert test_list_1[0].hourly_parameters.iloc[0, 0] == test_list_2[0].hourly_parameters.iloc[0, 0]


def test_fetch_historical(weather_provider_ecmwf):
    test_list = weather_provider_ecmwf.fetch_historical(["temperature_2m", "rain"])

    assert test_list is not None
    assert len(test_list) == 2
    assert test_list[0].hourly_parameters is not None
    assert list(test_list[0].hourly_parameters.columns) == ["temperature_2m", "rain"]


def test_fetch_historical_equivalence(weather_provider_ecmwf):
    test_list_1 = weather_provider_ecmwf.fetch_historical(["temperature_2m", "rain"])
    test_list_2 = weather_provider_ecmwf.fetch_historical(["temperature_2m", "rain"])

    assert test_list_1[0].hourly_parameters.iloc[0, 0] == test_list_2[0].hourly_parameters.iloc[0, 0]
