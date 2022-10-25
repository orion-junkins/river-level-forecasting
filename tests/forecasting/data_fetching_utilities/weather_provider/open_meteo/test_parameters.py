import pytest

from rlf.forecasting.data_fetching_utilities.weather_provider.open_meteo.parameters import get_hourly_parameters


@pytest.fixture
def valid_hourly_parameters():
    return [
        "temperature_2m",
        "relativehumidity_2m",
        "dewpoint_2m",
        "apparent_temperature",
        "pressure_msl",
        "surface_pressure",
        "precipitation",
        "rain",
        "snowfall",
        "cloudcover",
        "cloudcover_low",
        "cloudcover_mid",
        "cloudcover_high",
        "shortwave_radiation",
        "direct_radiation",
        "diffuse_radiation",
        "direct_normal_irradiance",
        "windspeed_10m",
        "windspeed_100m",
        "winddirection_10m",
        "winddirection_100m",
        "windgusts_10m",
        "et0_fao_evapotranspiration",
        "vapor_pressure_deficit",
        "soil_temperature_0_to_7cm",
        "soil_temperature_7_to_28cm",
        "soil_temperature_28_to_100cm",
        "soil_temperature_100_to_255cm",
        "soil_moisture_0_to_7cm",
        "soil_moisture_7_to_28cm",
        "soil_moisture_28_to_100cm",
        "soil_moisture_100_to_255cm"
    ]


def test_get_hourly_parameters_returns_valid_parameters(valid_hourly_parameters):
    assert get_hourly_parameters() == valid_hourly_parameters
