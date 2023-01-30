# %%
from rlf.forecasting.data_fetching_utilities.coordinate import Coordinate
from rlf.forecasting.data_fetching_utilities.weather_provider.api_weather_provider import APIWeatherProvider

# Declare coordinates
coordinates = [Coordinate(lon=-120.1, lat=44.1),
               Coordinate(lon=-120.2, lat=44.2),
               Coordinate(lon=-120.3, lat=44.3),
               Coordinate(lon=-120.4, lat=44.4),
               Coordinate(lon=-120.5, lat=44.5),
               Coordinate(lon=-120.6, lat=44.6),
               Coordinate(lon=-120.7, lat=44.7),
               Coordinate(lon=-120.8, lat=44.8),
               Coordinate(lon=-120.9, lat=44.9),
               Coordinate(lon=-121.0, lat=45.0)]

# Create an AWSWeatherProvider using the defined coordinates
weather_provider = APIWeatherProvider(coordinates)

# Fetch metadata for historical data
datums = weather_provider.fetch_historical(
    start_date="2010-01-01", end_date="2010-01-05")

# Inspect the 0th dataframe metadata
for datum in datums:
    print("(", datum.longitude, ",", datum.latitude, ")")

# %%
