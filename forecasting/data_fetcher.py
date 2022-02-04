from forecasting.general_utilities.time_utils import date_days_ago, unix_timestamp_days_ago
from forecasting.data_fetching_utilities.weather import *
from forecasting.data_fetching_utilities.level import get_historical_level
from forecasting.general_utilities.df_utils import *

class DataFetcher:
    """
    Basic data wrapper class to track all data relevant to an individual forecast sites. 
    This includes historical values for weather and level, and forecasted weather values to be used during inference.
    """
    def __init__(self, gauge_id, weather_sources) -> None:
        """
        Fetch all data for the particular site
        Args:
            gauge_id (str): USGS gauge id
            weather_sources (dict): dictionary of tuple:string containing ('lat', 'lon'):'path/to/data.json' for all weather data sources
        """
        self.gauge_id = gauge_id
        self.weather_locs = list(weather_sources.keys()) # list of (lat, lon) tuples
        self.weather_paths = list(weather_sources.values()) # list of data access paths

        # Level dataframes (indexed by 'datetime', columns = ['level'])
        self.historical_level = get_historical_level(self.gauge_id)
        self.recent_level = get_historical_level(self.gauge_id, start=date_days_ago(5))

        # Lists of weather dataframes (indexed by 'datetime', columns = ['temp', 'rain', etc.])
        self.historical_weather = get_all_historical_weather(self.weather_paths) 
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
        return merge(self.forecasted_weather)


    @property
    def all_historical_weather(self):
        """
        A single dataframe representing all historical weather from all locations.
        Returns:
            df: Merged dataframe.
        """
        return merge(self.historical_weather)


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

        return df_current
    

    @property
    def all_historical_data(self):
        """
        A single dataframe representing all data needed to perform training. Specifically, historical weather, and historical level.
        Returns:
            df: Merged dataframe
        """
        df_historical = self.all_historical_weather.join(self.historical_level, how='inner')
        return df_historical
        
        
    def update_for_inference(self):
        """
        Force an update to ensure inference data is up to date. Run at least hourly when forecasting.
        """
        self.recent_level = get_historical_level(self.gauge_id, start=date_days_ago(5))
        self.recent_weather = get_all_recent_weather(self.weather_locs, start=unix_timestamp_days_ago(5))
        self.forecasted_weather = get_all_forecasted_weather(self.weather_locs)