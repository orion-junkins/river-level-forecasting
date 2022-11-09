# %% Setup a weather provider
from rlf.forecasting.data_fetching_utilities.coordinate import Coordinate
from rlf.forecasting.data_fetching_utilities.weather_provider.weather_provider import WeatherProvider
from rlf.aws_dispatcher import AWSDispatcher

# Declare coordinates
coordinates = [Coordinate(lon=30.0, lat=52.5),
               Coordinate(lon=30.25, lat=52.5)]

# %% Create a historical dispatcher
historical_dispatcher = AWSDispatcher(bucket_name="historical-weather", directory_name="test-river")

# Create a historical weather provider using the defined coordinates
historical_weather_provider = WeatherProvider(coordinates, aws_dispatcher=historical_dispatcher)

# %% Update historical data in AWS
historical_weather_provider.update_historical_datums_in_aws(start_date="2000-01-01", end_date="2001-01-01")

# %% Given that AWS has historical data, fetch it as a list of dataframes
dfs = historical_weather_provider.fetch_historical()

# %% Inspect the 0th dataframe
dfs[0]

# %% Fetch a subset of columns only
dfs = historical_weather_provider.fetch_historical(columns=['temperature_2m', 'dewpoint_2m'])
# %% Inspect the 0th dataframe
dfs[0]

# %% Create a current dispatcher
current_dispatcher = AWSDispatcher(bucket_name="current-weather-data", directory_name="test-river")

# Create a current weather provider using the defined coordinates
current_weather_provider = WeatherProvider(coordinates, aws_dispatcher=current_dispatcher)

# %% Update current data in AWS
current_weather_provider.update_current_datums_in_aws()

# %% Given that AWS has current data, fetch it as a list of dataframes
dfs = current_weather_provider.fetch_current()

# %% Inspect the 0th dataframe
dfs[0]

# %% Fetch a subset of columns only
dfs = current_weather_provider.fetch_current(columns=['temperature_2m', 'dewpoint_2m'])
# %% Inspect the 0th dataframe
dfs[0]
# %%
