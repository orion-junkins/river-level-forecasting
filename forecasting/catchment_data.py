from data.weather_locations import all_weather_locs
from forecasting.general_utilities.time_utils import date_days_ago, unix_timestamp_days_ago
from forecasting.data_fetching_utilities.weather import *
from forecasting.data_fetching_utilities.level import get_historical_level

class CatchmentData:
    def __init__(self, name, usgs_gauge_id, load_historical=True, level_forecast_url=None, weather_locs=None, window_size=40) -> None:
        self.name = name
        self.usgs_gauge_id = str(usgs_gauge_id)
        self.level_forecast_url = level_forecast_url

        if weather_locs == None:
            self.weather_locs = all_weather_locs[name]
        else:
            self.weather_locs = weather_locs

        self.window_size = window_size

        if load_historical:
            self.historical_level = get_historical_level(self.usgs_gauge_id)
            self.historical_weather = fetch_all_historical_weather(self.weather_locs, self.name) 
        
        self.recent_level = get_historical_level(self.usgs_gauge_id, start=date_days_ago(self.window_size))
        self.recent_weather = fetch_all_recent_weather(self.weather_locs, start=unix_timestamp_days_ago(self.window_size))
        self.forecasted_weather = fetch_all_forecasted_weather(self.weather_locs)


    @property
    def all_current_data(self):
        """
        """
        all_current_data = []
        for df_recent, df_forecasted in zip(self.recent_weather, self.forecasted_weather):
            weather_frames = [df_recent, df_forecasted]
            df = pd.concat(weather_frames) # Combine recent and forecasted weather into a single df
            df = handle_missing_data(df)
            all_current_data.append(df)
        
        return (all_current_data, self.recent_level)
    

    @property
    def all_historical_data(self):
        """
        """
        return (self.historical_weather, self.historical_level)
        
        
    def update_for_inference(self):
        """
        Force an update to ensure inference data is up to date. Run at least hourly when forecasting.
        """
        self.recent_level = get_historical_level(self.usgs_gauge_id, start=date_days_ago(self.window_size))
        self.recent_weather = fetch_all_recent_weather(self.weather_locs, start=unix_timestamp_days_ago(self.window_size))
        self.forecasted_weather = fetch_all_forecasted_weather(self.weather_locs)