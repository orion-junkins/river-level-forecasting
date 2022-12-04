# Demo of the APIWeatherProvider class. Demonstrates fetching historical and current data. Also demonstrates fetching column subsets.
# Expects that data exists in AWS for the specified coordinates and timestamps.
# Run the scripts in `scripts/upload/weather/` to upload weather to AWS.
# %% ------------------------------
# Imports
from rlf.forecasting.data_fetching_utilities.coordinate import Coordinate
from rlf.aws_dispatcher import AWSDispatcher
from rlf.forecasting.data_fetching_utilities.weather_provider.aws_weather_provider import AWSWeatherProvider

# Declare coordinates
coordinates = [Coordinate(lon=-120.75, lat=44.25), Coordinate(lon=-121.5, lat=44.25)]

# Create an AWSDispatcher
aws_dispatcher = AWSDispatcher(bucket_name="all-weather-data", directory_name="open-meteo")

# Create an AWSWeatherProvider using the defined coordinates and dispatcher
weather_provider = AWSWeatherProvider(coordinates, aws_dispatcher=aws_dispatcher)


# %% ------------------------------
# Fetch historical data
datums = weather_provider.fetch_historical(start_date="2010-01-01", end_date="2010-01-05")

# View the 0th fetched datum's hourly parameters
datums[0].hourly_parameters


# %% ------------------------------
# Fetch historical data for a subset of columns only
datums = weather_provider.fetch_historical(columns=['temperature_2m', 'dewpoint_2m'])

# Inspect the 0th dataframe
datums[0].hourly_parameters


# %% ------------------------------
# Supply a timestamp to the AWSWeatherProvider
weather_provider.set_timestamp("22-12-03_20-09")

# Fetch 'current' data logged at the given timestamp
datums = weather_provider.fetch_current()

# Inspect the 0th dataframe
datums[0].hourly_parameters


# %% ------------------------------
# Fetch a subset of columns only
datums = weather_provider.fetch_current(columns=['temperature_2m', 'dewpoint_2m'])

# Inspect the 0th dataframe
datums[0].hourly_parameters
