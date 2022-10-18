import numpy as np
import pandas as pd
import pytest

from forecasting.catchment_data import CatchmentData


def level_df(num_samples=100):
    return pd.DataFrame(np.random.randint(0, 100, size=(num_samples, 2)), columns=["datetime", "level"])


class FakeLevelProvider():
    def __init__(self) -> None:
        pass

    def fetch_recent_level(self, samples_to_fetch):
        return level_df(samples_to_fetch)

    def fetch_historical_level(self):
        return level_df()


@pytest.fixture
def fake_level_provider():
    return FakeLevelProvider()


def weather_df(num_samples):
    return pd.DataFrame(np.random.randint(0, 100, size=(num_samples, 4)), columns=list('ABCD'))


def weather_dfs(num_samples, num_dfs):
    dfs = []
    for _ in range(num_dfs):
        dfs.append(weather_df(num_samples))
    return dfs


class FakeWeatherProvider:
    def __init__(self) -> None:
        self.num_locs = 12

    def fetch_current_weather(self, samples_to_fetch):
        return weather_dfs(samples_to_fetch, self.num_locs)

    def fetch_historical_weather(self):
        return weather_dfs(234, self.num_locs)


@pytest.fixture
def fake_weather_provider():
    return FakeWeatherProvider()


def test_num_data_sets(fake_weather_provider, fake_level_provider):
    catchment = CatchmentData("test_catchment", fake_weather_provider, fake_level_provider)
    num_data_sets = catchment.num_data_sets

    assert (fake_weather_provider.num_locs == num_data_sets)


def test_all_current_correct_num_samples(fake_weather_provider, fake_level_provider):
    num_samples = 123
    catchment = CatchmentData("test_catchment", fake_weather_provider, fake_level_provider, num_recent_samples=num_samples)

    weather_dfs, level_df = catchment.all_current

    assert (len(level_df) == num_samples)
    for weather_df in weather_dfs:
        assert (len(weather_df) == num_samples)
