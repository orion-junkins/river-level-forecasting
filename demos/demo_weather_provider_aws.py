# %% Setup a weather provider
from rlf.forecasting.data_fetching_utilities.coordinate import Coordinate
from rlf.forecasting.data_fetching_utilities.weather_provider.weather_provider import WeatherProvider
from rlf.aws_dispatcher import AWSDispatcher

# Declare coordinates
coordinates = [Coordinate(lon=30.0, lat=52.5),
               Coordinate(lon=30.25, lat=52.5)]

# %% Create a historical dispatcher
aws_dispatcher = AWSDispatcher(bucket_name="historical-weather", directory_name="test-river")

# Create a historical weather provider using the defined coordinates
weather_provider = WeatherProvider(coordinates, aws_dispatcher=aws_dispatcher)

# %% Update historical data in AWS
weather_provider.update_historical_datums_in_aws(start_date="2000-01-01", end_date="2001-01-01")

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
