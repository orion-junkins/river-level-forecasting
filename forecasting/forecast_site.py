from forecasting.time_utils import date_days_ago, unix_timestamp_days_ago
from forecasting.data_fetching_utilities.weather import *
from forecasting.data_fetching_utilities.level import get_historical_level

class ForecastSite:
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
        self.weather_locs = list(weather_sources.keys())
        self.weather_paths = list(weather_sources.values())

        # Level dataframes (indexed by 'datetime', columns = ['level'])
        #self.hisotrical_level = get_historical_level(self.gauge_id)
        self.recent_level = get_historical_level(self.gauge_id, start=date_days_ago(5))

        # Lists of weather dataframes (indexed by 'datetime', columns = ['temp', 'rain', etc.])
        self.historical_weather = get_all_historical_weather(self.weather_paths) 
        self.recent_weather = get_all_recent_weather(self.weather_locs, start=unix_timestamp_days_ago(5))
        self.forecasted_weather = get_all_forecasted_weather(self.weather_locs)
    

    def update_for_inference(self):
        self.recent_level = get_historical_level(self.gauge_id, start=date_days_ago(5))
        self.recent_weather = get_all_recent_weather(self.weather_locs, start=date_days_ago(5))
        self.forecasted_weather = get_all_forecasted_weather(self.weather_locs)
