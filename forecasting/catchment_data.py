from data.weather_locations import all_weather_locs
from forecasting.general_utilities.time_utils import date_days_ago, unix_timestamp_days_ago
from forecasting.data_fetching_utilities.weather import *
from forecasting.data_fetching_utilities.level import get_historical_level
from forecasting.general_utilities.df_utils import *

class CatchmentData:
    def __init__(self, name, usgs_gauge_id, level_forecast_url=None, weather_locs=None, window_size=40) -> None:
        self.name = name
        self.usgs_gauge_id = str(usgs_gauge_id)
        self.level_forecast_url = level_forecast_url

        if weather_locs == None:
            self.weather_locs = all_weather_locs[name]
        else:
            self.weather_locs = weather_locs

        self.window_size = window_size

        # Level dataframes (indexed by 'datetime', columns = ['level'])
        self.historical_level = get_historical_level(self.usgs_gauge_id)
        self.recent_level = get_historical_level(self.usgs_gauge_id, start=date_days_ago(self.window_size))

        # Lists of weather dataframes (indexed by 'datetime', columns = ['temp', 'rain', etc.])
        self.historical_weather = fetch_all_historical_weather(self.weather_locs, self.name) 
        self.recent_weather = fetch_all_recent_weather(self.weather_locs, start=unix_timestamp_days_ago(self.window_size))
        self.forecasted_weather = fetch_all_forecasted_weather(self.weather_locs)


    @property
    def all_recent_weather(self):
        """
        A single dataframe representing all recent (5da) weather from all locations.
        TODO: Ensure no clashes between column names
        Returns:
            df: Merged dataframe.
        """
        return merge(self.recent_weather)
    

    @property
    def all_forecasted_weather(self):
        """
        A single dataframe representing all forecasted weather from all locations.
        Returns:
            df: Merged dataframe.
        """
        df = merge(self.forecasted_weather)
        df = handle_missing_data(df)
        return df


    @property
    def all_historical_weather(self):
        """
        A single dataframe representing all historical weather from all locations.
        Returns:
            df: Merged dataframe.
        """
        df = merge(self.historical_weather)
        df = handle_missing_data(df)
        return df


    @property
    def all_current_data(self):
        """
        """
        all_current_data = []
        for df_recent, df_forecasted in zip(self.recent_weather, self.forecasted_weather):
            weather_frames = [df_recent, df_forecasted]
            df_weather = pd.concat(weather_frames) # Combine recent and forecasted weather into a single df
            df = pd.concat([df_weather, self.recent_level], axis=1, join='inner') # Add level data
            df = handle_missing_data(df)
            all_current_data.append(df)
        
        return all_current_data
    

    @property
    def all_historical_data(self):
        """
        """
        all_historical_data = []
        for df_weather in self.historical_weather:
            df = pd.concat([df_weather, self.historical_level], axis=1, join='inner') # Add level data
            df = handle_missing_data(df)
            all_historical_data.append(df)
        return all_historical_data
        
        
    def update_for_inference(self):
        """
        Force an update to ensure inference data is up to date. Run at least hourly when forecasting.
        """
        self.recent_level = get_historical_level(self.usgs_gauge_id, start=date_days_ago(self.window_size))
        self.recent_weather = fetch_all_recent_weather(self.weather_locs, start=unix_timestamp_days_ago(self.window_size))
        self.forecasted_weather = fetch_all_forecasted_weather(self.weather_locs)