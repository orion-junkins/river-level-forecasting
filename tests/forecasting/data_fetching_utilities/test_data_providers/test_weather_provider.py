import pytest
import pandas as pd
from rlf.forecasting.data_fetching_utilities.data_providers.weather_provider import WeatherProvider
from rlf.forecasting.data_fetching_utilities.data_providers.datum import Datum


class TestWeatherProvider():

    def test_initialization(self):
        try:
            WeatherProvider(locations=[(5.55, 5.55), (5.55, 5.55)])
        except NameError:
            pytest.fail("WeatherProvider failed to initialize")

    def test_has_locations(self):
        provider = WeatherProvider([(5.55, 5.55), (5.55, 5.55)])
        assert provider.locations == [(5.55, 5.55), (5.55, 5.55)]

    # returns a list of Datum objects
    def test_fetch_historical_weather_returns_list_of_datums(self):
        provider = WeatherProvider([(5.55, 5.55), (5.55, 5.55)])
        datums = provider.fetch_historical_weather()
        assert isinstance(datums, list)
        assert isinstance(datums[0], Datum)
