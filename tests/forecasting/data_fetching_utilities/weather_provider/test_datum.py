import pandas as pd
import pytest

from rlf.forecasting.data_fetching_utilities.weather_provider.weather_datum import WeatherDatum


@pytest.fixture
def datum():
    return WeatherDatum(longitude=0, latitude=0, elevation=0, utc_offset_seconds=0,
                        timezone="UTC", hourly_parameters={})


def test_get_hourly_parameters_returns_dataframe(datum):
    assert isinstance(datum.get_data_frame(), pd.DataFrame)


def test_get_hourly_parameters_returns_dataframe_with_correct_columns(datum):
    assert datum.get_data_frame().columns.tolist() == list(
        datum.hourly_parameters.keys())


def test_returns_longitude(datum):
    assert datum.longitude == 0


def test_returns_latitude(datum):
    assert datum.latitude == 0


def test_returns_elevation(datum):
    assert datum.elevation == 0


def test_returns_utc_offset_seconds(datum):
    assert datum.utc_offset_seconds == 0


def test_returns_timezone(datum):
    assert datum.timezone == "UTC"
