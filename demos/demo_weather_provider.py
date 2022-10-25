
# %%
from rlf.forecasting.data_fetching_utilities.coordinate import Coordinate
from rlf.forecasting.data_fetching_utilities.weather_provider.weather_provider import WeatherProvider
from rlf.forecasting.data_fetching_utilities.weather_provider.open_meteo.open_meteo_adapter import OpenMeteoAdapter

coordinates = [Coordinate(lat=52.52, lon=30),
               Coordinate(lat=52.52, lon=30.1)]

weather_provider = WeatherProvider(
    coordinates=coordinates, api_adapter=OpenMeteoAdapter())
list_of_points_with_weather = weather_provider.fetch_historical_weather(
    start_date="2021-01-01", end_date="2021-01-01")

# %%
print(type(list_of_points_with_weather[0]))
print(len(list_of_points_with_weather))
for datum in list_of_points_with_weather:
    print(datum)

# %%
