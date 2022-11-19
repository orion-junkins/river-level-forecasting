# %% Setup a weather provider
from rlf.forecasting.data_fetching_utilities.coordinate import Coordinate
from rlf.forecasting.data_fetching_utilities.weather_provider.weather_provider import WeatherProvider

# Declare coordinates
coordinates = [Coordinate(lon=30.0, lat=52.5),
               Coordinate(lon=30.25, lat=52.5)]

# %%
# Create a historical weather provider using the defined coordinates
weather_provider = WeatherProvider(coordinates)

# %% Given that AWS has historical data, fetch it as a list of dataframes
dfs = weather_provider.fetch_historical()

# %% Inspect the 0th dataframe
dfs[0]

# %% Fetch a subset of columns only
dfs = weather_provider.fetch_historical(columns=['temperature_2m', 'dewpoint_2m'])
# %% Inspect the 0th dataframe
dfs[0]


# %% Given that AWS has current data, fetch it as a list of dataframes
dfs = weather_provider.fetch_current()

# %% Inspect the 0th dataframe
dfs[0]

# %% Fetch a subset of columns only
dfs = weather_provider.fetch_current(columns=['temperature_2m', 'dewpoint_2m'])
# %% Inspect the 0th dataframe
dfs[0]
# %%
