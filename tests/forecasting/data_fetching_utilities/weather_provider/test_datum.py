from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import pytest

from rlf.forecasting.data_fetching_utilities.weather_provider.weather_datum import WeatherDatum


def dummy_hourly_dt_index(num_samples):
    date_today = datetime(2021, 1, 1)
    hours = pd.date_range(date_today, date_today + timedelta(hours=num_samples), freq='H')
    hours = hours[0:num_samples]
    return hours


def weather_df(num_samples):
    hours = dummy_hourly_dt_index(num_samples)
    np.random.seed(seed=1)
    data_1 = np.random.randint(1, high=100, size=len(hours))
    data_2 = np.random.randint(1, high=100, size=len(hours))
    df = pd.DataFrame({'datetime': hours, 'weather_attr_1': data_1, 'weather_attr_2': data_2})
    df = df.set_index('datetime')
    return df


@pytest.fixture
def datum():
    return WeatherDatum(longitude=0, latitude=0, elevation=0, utc_offset_seconds=0,
                        timezone="Fake Time Zone", hourly_units="Fake Units", hourly_parameters=weather_df(10))


def test_meta_data(datum):
    meta_data = datum.meta_data
    assert meta_data["longitude"] == 0
    assert meta_data["latitude"] == 0
    assert meta_data["elevation"] == 0
    assert meta_data["utc_offset_seconds"] == 0
    assert meta_data["timezone"] == "Fake Time Zone"


def test_returns_longitude(datum):
    assert datum.longitude == 0


def test_returns_latitude(datum):
    assert datum.latitude == 0


def test_returns_elevation(datum):
    assert datum.elevation == 0


def test_returns_utc_offset_seconds(datum):
    assert datum.utc_offset_seconds == 0


def test_returns_timezone(datum):
    assert datum.timezone == "Fake Time Zone"
