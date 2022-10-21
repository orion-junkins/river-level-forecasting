import pandas as pd
import pytest

from rlf.forecasting.data_fetching_utilities.weather_provider.datum import Datum


@pytest.fixture
def datum():
    return Datum(longitude=0, latitude=0, elevation=0, utc_offset_seconds=0,
                 timezone="UTC", hourly_parameters={})


def test_get_hourly_parameters_returns_dict(datum):
    assert isinstance(datum.get_hourly_parameters(
        as_pandas_data_frame=False), dict)


def test_get_hourly_parameters_returns_dataframe(datum):
    assert isinstance(datum.get_hourly_parameters(
        as_pandas_data_frame=True), pd.DataFrame)


def test_get_hourly_parameters_returns_dataframe_with_correct_columns(datum):
    assert datum.get_hourly_parameters(
        as_pandas_data_frame=True).columns.tolist() == list(datum.hourly_parameters.keys())


def test_returns_longitude(datum):
    assert datum.get_longitude() == 0


def test_returns_latitude(datum):
    assert datum.get_latitude() == 0


def test_returns_elevation(datum):
    assert datum.get_elevation() == 0


def test_returns_utc_offset_seconds(datum):
    assert datum.get_utc_offset_seconds() == 0


def test_returns_timezone(datum):
    assert datum.get_timezone() == "UTC"
