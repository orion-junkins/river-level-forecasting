
# %%
from rlf.forecasting.data_fetching_utilities.coordinate import Coordinate
from rlf.forecasting.data_fetching_utilities.weather_provider.weather_provider import WeatherProvider

coordinates = [Coordinate(lat=52.52, lon=13.40),
               Coordinate(lat=52.52, lon=13.40)]

weather_provider = WeatherProvider(coordinates=coordinates)
wx = weather_provider.fetch_historical_weather(
    start_date="2021-01-01", end_date="2021-01-02")
print(type(wx))

# %%
for datum in wx:
    print(datum.elevation)

# %%
