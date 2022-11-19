import pytest

from rlf.forecasting.catchment_data import CatchmentData
from fake_providers import FakeLevelProvider, FakeWeatherProvider


@pytest.fixture
def fake_level_provider():
    return FakeLevelProvider()


@pytest.fixture
def fake_weather_provider():
    return FakeWeatherProvider()


def test_num_weather_datasets(fake_weather_provider, fake_level_provider):
    catchment = CatchmentData("test_catchment", fake_weather_provider, fake_level_provider)
    num_weather_datasets = catchment.num_weather_datasets

    assert (fake_weather_provider.num_locs == num_weather_datasets)


def test_all_current_correct_num_samples(fake_weather_provider, fake_level_provider):
    num_samples = 123
    catchment = CatchmentData("test_catchment", fake_weather_provider, fake_level_provider, num_recent_samples=num_samples)

    weather_dfs, level_df = catchment.all_current

    assert (len(level_df) == num_samples)
