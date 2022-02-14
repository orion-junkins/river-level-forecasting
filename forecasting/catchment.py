from data.weather_locations import all_weather_locs
from forecasting.general_utilities.time_utils import date_days_ago, unix_timestamp_days_ago
from forecasting.data_fetching_utilities.weather import *
from forecasting.data_fetching_utilities.level import get_historical_level
from forecasting.general_utilities.df_utils import *
import os
class Catchment:
    def __init__(self, name, usgs_gauge_id, level_forecast_url=None, weather_locs=None) -> None:
        self.name = name
        self.usgs_gauge_id = str(usgs_gauge_id)
        self.level_forecast_url = level_forecast_url

        if weather_locs == None:
            self.weather_locs = all_weather_locs[name]
        else:
            self.weather_locs = weather_locs


        # Level dataframes (indexed by 'datetime', columns = ['level'])
        self.historical_level = get_historical_level(self.usgs_gauge_id)
        self.recent_level = get_historical_level(self.usgs_gauge_id, start=date_days_ago(5))

        # Lists of weather dataframes (indexed by 'datetime', columns = ['temp', 'rain', etc.])
        self.historical_weather = get_all_historical_weather(self.weather_locs, os.path.join("data", "historical", self.name)) 
        self.recent_weather = get_all_recent_weather(self.weather_locs, start=unix_timestamp_days_ago(5))
        self.forecasted_weather = get_all_forecasted_weather(self.weather_locs)


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
        A single dataframe representing all data needed to perform inference. Specifically, recent weather, recent level, and forecasted level.
        Returns:
            df: Merged dataframe
        """
        df_recent = pd.concat([self.all_recent_weather, self.recent_level], axis=1, join='inner') #TODO: Revise to use join
        all_current_frames = [df_recent, self.all_forecasted_weather]
        df_current = pd.concat(all_current_frames) 
        df_current = handle_missing_data(df_current)
        return df_current
    

    @property
    def all_historical_data(self):
        """
        A single dataframe representing all data needed to perform training. Specifically, historical weather, and historical level.
        Returns:
            df: Merged dataframe
        """
        df = self.all_historical_weather.join(self.historical_level, how='inner')
        df = handle_missing_data(df)
        return df
        
        
    def update_for_inference(self):
        """
        Force an update to ensure inference data is up to date. Run at least hourly when forecasting.
        """
        self.recent_level = get_historical_level(self.usgs_gauge_id, start=date_days_ago(5))
        self.recent_weather = get_all_recent_weather(self.weather_locs, start=unix_timestamp_days_ago(5))
        self.forecasted_weather = get_all_forecasted_weather(self.weather_locs)