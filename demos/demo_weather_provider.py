
# %%
from rlf.forecasting.data_fetching_utilities.coordinate import Coordinate
from rlf.forecasting.data_fetching_utilities.weather_provider.weather_provider import WeatherProvider

coordinates = [Coordinate(lat=52.52, lon=30),
               Coordinate(lat=52.52, lon=30.1), ]

weather_provider = WeatherProvider(coordinates=coordinates)
wx = weather_provider.fetch_historical_weather(
    start_date="2021-01-01", end_date="2021-01-02")

# %%
for datum in wx:
    print(datum)

# %%
