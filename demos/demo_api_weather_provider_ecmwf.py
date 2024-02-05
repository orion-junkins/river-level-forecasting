# Demo of the APIWeatherProvider class. Demonstrates fetching historical and current data. Also demonstrates fetching column subsets.
# %% ------------------------------
# Imports
from rlf.forecasting.data_fetching_utilities.coordinate import Coordinate
from rlf.forecasting.data_fetching_utilities.weather_provider.api_weather_provider_ecmwf import APIWeatherProviderECMWF

# Declare coordinates
coordinates = [Coordinate(lon=-121.5, lat=47.3),
                   Coordinate(lon=-121.3, lat=47.4)]

# Create an AWSWeatherProvider using the defined coordinates
weather_provider = APIWeatherProviderECMWF(coordinates)


# %% ------------------------------
# Fetch historical data
datums = weather_provider.fetch_historical(start_date="2018-01-01", end_date="2018-01-05")

# Inspect the 0th dataframe
datums[0].hourly_parameters


# %% ------------------------------
# Fetch historical data for a subset of columns only
datums = weather_provider.fetch_historical(columns=['temperature_2m', 'dewpoint_2m'])

# Inspect the 0th dataframe
datums[0].hourly_parameters


# %% ------------------------------
# Fetch current data
datums = weather_provider.fetch_current()

# Inspect the 0th dataframe
datums[0].hourly_parameters


# %% ------------------------------
# Fetch a subset of columns only
datums = weather_provider.fetch_current(columns=['temperature_2m', 'dewpoint_2m'])

# Inspect the 0th dataframe
datums[0].hourly_parameters
