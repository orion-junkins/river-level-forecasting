from typing import List

import pytest

from rlf.aws_dispatcher import AWSDispatcher
from rlf.forecasting.data_fetching_utilities.coordinate import Coordinate
from rlf.forecasting.data_fetching_utilities.weather_provider.aws_weather_provider import AWSWeatherProvider

# It is expected that datums will be stored for this timestamp in the bucket specified for the coordinates specified.
CURRENT_TESTING_TIMESTAMP = "23-01-31_07-42"


@pytest.fixture
def aws_dispatcher() -> AWSDispatcher:
    return AWSDispatcher("all-weather-data", "testing")


@pytest.fixture
def coordinates() -> List[Coordinate]:
    return [Coordinate(lon=-120.8, lat=44.2), Coordinate(lon=-121.8, lat=44.3)]


@ pytest.fixture
def weather_provider(aws_dispatcher, coordinates) -> AWSWeatherProvider:
    return AWSWeatherProvider(
        coordinates=coordinates, aws_dispatcher=aws_dispatcher)


@pytest.mark.aws
@pytest.mark.slow
def test_fetch_historical(weather_provider):
    expected_columns = [
        'apparent_temperature',
        'cloudcover',
        'cloudcover_high',
        'cloudcover_low',
        'cloudcover_mid',
        'dewpoint_2m',
        'diffuse_radiation',
        'direct_normal_irradiance',
        'direct_radiation',
        'et0_fao_evapotranspiration',
        'precipitation',
        'pressure_msl',
        'rain',
        'relativehumidity_2m',
        'shortwave_radiation',
        'snowfall',
        'soil_moisture_level_1',
        'soil_moisture_level_2',
        'soil_moisture_level_3',
        'soil_moisture_level_4',
        'soil_temperature_level_1',
        'soil_temperature_level_2',
        'soil_temperature_level_3',
        'soil_temperature_level_4',
        'surface_pressure',
        'temperature_2m',
        'vapor_pressure_deficit',
        'winddirection_100m',
        'winddirection_10m',
        'windgusts_10m',
        'windspeed_100m',
        'windspeed_10m'
    ]
    weather_datums = weather_provider.fetch_historical()
    assert len(weather_datums) == len(weather_provider.coordinates)
    for weather_datum in weather_datums:
        assert weather_datum.hourly_parameters.index.dtype == "datetime64[ns, UTC]"
        assert expected_columns == sorted(list(weather_datum.hourly_parameters.columns))


@pytest.mark.aws
@pytest.mark.slow
def test_fetch_current(weather_provider):
    expected_columns = [
        'apparent_temperature',
        'cape',
        'cloudcover',
        'cloudcover_high',
        'cloudcover_low',
        'cloudcover_mid',
        'dewpoint_2m',
        'et0_fao_evapotranspiration',
        'evapotranspiration',
        'freezinglevel_height',
        'lifted_index',
        'precipitation',
        'pressure_msl',
        'relativehumidity_2m',
        'snow_depth',
        'snowfall',
        'soil_moisture_level_1',
        'soil_moisture_level_2',
        'soil_moisture_level_3',
        'soil_moisture_level_4',
        'soil_temperature_level_1',
        'soil_temperature_level_2',
        'soil_temperature_level_3',
        'soil_temperature_level_4',
        'surface_pressure',
        'temperature_2m',
        'vapor_pressure_deficit',
        'visibility',
        'winddirection_10m',
        'winddirection_80m',
        'windgusts_10m',
        'windspeed_10m',
        'windspeed_80m'
    ]

    weather_provider.set_timestamp(CURRENT_TESTING_TIMESTAMP)
    weather_datums = weather_provider.fetch_current()
    assert len(weather_datums) == len(weather_provider.coordinates)
    for weather_datum in weather_datums:
        assert weather_datum.hourly_parameters.index.dtype == "datetime64[ns, UTC]"
        assert expected_columns == sorted(list(weather_datum.hourly_parameters.columns))


@pytest.mark.aws
@pytest.mark.slow
def test_fetch_current_with_consistent_columns(weather_provider):
    expected_columns = [
        'soil_moisture_level_1',
        'soil_moisture_level_2',
        'soil_moisture_level_3',
        'soil_moisture_level_4',
        'soil_temperature_level_1',
        'soil_temperature_level_2',
        'soil_temperature_level_3',
        'soil_temperature_level_4',
    ]

    requested_columns = [
        'soil_moisture_level_1',
        'soil_moisture_level_2',
        'soil_moisture_level_3',
        'soil_moisture_level_4',
        'soil_temperature_level_1',
        'soil_temperature_level_2',
        'soil_temperature_level_3',
        'soil_temperature_level_4',
    ]

    weather_provider.set_timestamp(CURRENT_TESTING_TIMESTAMP)
    weather_datums = weather_provider.fetch_current(columns=requested_columns)

    for weather_datum in weather_datums:
        assert expected_columns == sorted(list(weather_datum.hourly_parameters.columns))


@pytest.mark.aws
@pytest.mark.slow
def test_fetch_historical_with_consistent_columns(weather_provider):
    expected_columns = [
        'soil_moisture_level_1',
        'soil_moisture_level_2',
        'soil_moisture_level_3',
        'soil_moisture_level_4',
        'soil_temperature_level_1',
        'soil_temperature_level_2',
        'soil_temperature_level_3',
        'soil_temperature_level_4',
    ]

    requested_columns = [
        'soil_moisture_level_1',
        'soil_moisture_level_2',
        'soil_moisture_level_3',
        'soil_moisture_level_4',
        'soil_temperature_level_1',
        'soil_temperature_level_2',
        'soil_temperature_level_3',
        'soil_temperature_level_4',
    ]

    weather_provider.set_timestamp(CURRENT_TESTING_TIMESTAMP)
    weather_datums = weather_provider.fetch_historical(columns=requested_columns)

    for weather_datum in weather_datums:
        assert expected_columns == sorted(list(weather_datum.hourly_parameters.columns))
