import pytest
import pandas as pd
from forecasting.data_fetching_utilities.data_providers.weather_provider import WeatherProvider


class TestWeatherProvider():

    def test_initialization(self):
        try:
            WeatherProvider(locations=[(5.55, 5.55), (5.55, 5.55)])
        except NameError:
            pytest.fail("WeatherProvider failed to initialize")

    def test_has_locations(self):
        provider = WeatherProvider([(5.55, 5.55), (5.55, 5.55)])
        assert provider.locations == [(5.55, 5.55), (5.55, 5.55)]

    def test_fetch_historical_weather(self):
        provider = WeatherProvider([(5.55, 5.55), (5.55, 5.55)])
        provider.fetch_historical_weather()
        assert isinstance(provider.fetch_historical_weather(), pd.DataFrame)
