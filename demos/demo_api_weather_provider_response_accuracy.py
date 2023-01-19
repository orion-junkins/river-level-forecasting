# %%
from rlf.forecasting.data_fetching_utilities.coordinate import Coordinate
from rlf.forecasting.data_fetching_utilities.weather_provider.api_weather_provider import APIWeatherProvider

# Declare coordinates
coordinates = [Coordinate(lon=-120.1, lat=44.1),
               Coordinate(lon=-121.2, lat=44.2),
               Coordinate(lon=-121.3, lat=44.3),
               Coordinate(lon=-121.4, lat=45.4),
               Coordinate(lon=-121.5, lat=45.5),
               Coordinate(lon=-122.6, lat=45.6),
               Coordinate(lon=-122.7, lat=45.7),
               Coordinate(lon=-122.8, lat=46.8),
               Coordinate(lon=-122.9, lat=46.9),
               Coordinate(lon=-123.0, lat=47.0)]

# Create an AWSWeatherProvider using the defined coordinates
weather_provider = APIWeatherProvider(coordinates)

# Fetch metadata for historical data
datums = weather_provider.fetch_historical(
    start_date="2010-01-01", end_date="2010-01-05")

# Inspect the 0th dataframe metadata
for datum in datums:
    print("(", datum.longitude, ",", datum.latitude, ")")

# %%
