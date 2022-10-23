import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def dummy_hourly_dt_index(num_samples):
    date_today = datetime(2021, 1, 1)
    hours = pd.date_range(date_today, date_today + timedelta(hours=num_samples), freq='H')
    hours = hours[0:num_samples]
    return hours


def level_df(num_samples=100):
    hours = dummy_hourly_dt_index(num_samples)
    np.random.seed(seed=1)
    data = np.random.randint(1, high=100, size=len(hours))
    df = pd.DataFrame({'datetime': hours, 'level': data})
    df = df.set_index('datetime')
    return df


class FakeLevelProvider():
    def __init__(self, num_historical_samples=101) -> None:
        self.num_historical_samples = num_historical_samples

    def fetch_recent_level(self, samples_to_fetch):
        return level_df(samples_to_fetch)

    def fetch_historical_level(self):
        return level_df(self.num_historical_samples)


def weather_df(num_samples):
    hours = dummy_hourly_dt_index(num_samples)
    np.random.seed(seed=1)
    data_1 = np.random.randint(1, high=100, size=len(hours))
    data_2 = np.random.randint(1, high=100, size=len(hours))
    df = pd.DataFrame({'datetime': hours, 'weather_attr_1': data_1, 'weather_attr_2': data_2})
    df = df.set_index('datetime')
    return df


def weather_dfs(num_samples, num_dfs):
    dfs = []
    for _ in range(num_dfs):
        dfs.append(weather_df(num_samples))
    return dfs


class FakeWeatherProvider:
    def __init__(self, num_locs=12, num_historical_samples=100) -> None:
        self.num_locs = num_locs
        self.num_historical_samples = num_historical_samples

    def fetch_current_weather(self, samples_to_fetch):
        return weather_dfs(samples_to_fetch, self.num_locs)

    def fetch_historical_weather(self):
        return weather_dfs(self.num_historical_samples, self.num_locs)
