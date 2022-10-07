from data.weather_locations import all_weather_locs
from forecasting.general_utilities.time_utils import date_days_ago, unix_timestamp_days_ago
from forecasting.data_fetching_utilities.weather import *
from forecasting.data_fetching_utilities.level import get_historical_level

class CatchmentData:
    def __init__(self, name, usgs_gauge_id, weather_locs=None, window_size=40) -> None:
        self.name = name
        self.usgs_gauge_id = str(usgs_gauge_id)

        if weather_locs == None:
            self.weather_locs = all_weather_locs[name]
        else:
            self.weather_locs = weather_locs

        self.window_size = window_size


    # CURRENT
    @property
    def all_current(self):
        recent_level = get_historical_level(self.usgs_gauge_id, start=date_days_ago(self.window_size))
        recent_weather = fetch_all_recent_weather(self.weather_locs, start=unix_timestamp_days_ago(self.window_size))
        forecasted_weather = fetch_all_forecasted_weather(self.weather_locs)
        
        current_weather = []
        for df_recent, df_forecasted in zip(recent_weather, forecasted_weather):
            weather_frames = [df_recent, df_forecasted]
            df = pd.concat(weather_frames) # Combine recent and forecasted weather into a single df
            df = handle_missing_data(df)
            current_weather.append(df)
        
        return (current_weather, recent_level)
    

    # HISTORICAL
    @property
    def historical_level(self):
        return get_historical_level(self.usgs_gauge_id)

    @property
    def historical_weather(self):
        return fetch_all_historical_weather(self.weather_locs, self.name) 

    @property
    def all_historical(self):
        return (self.historical_weather, self.historical_level)