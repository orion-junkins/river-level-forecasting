from data.weather_locations import all_weather_locs
from forecasting.data_providers.weather_provider import WeatherProvider
from forecasting.data_providers.level_provider import LevelProvider

class CatchmentData:
    def __init__(self, catchment_name, usgs_gauge_id, num_recent_samples=40*24) -> None:
        self.weather_provider = WeatherProvider(all_weather_locs[catchment_name])
        self.level_provider = LevelProvider(str(usgs_gauge_id))
        self.num_recent_samples = num_recent_samples

        self._all_current = None 
        self._all_historical = None
 
    @property
    def all_current(self):
        if self._all_current == None:
            self._get_all_current()
        
        return self._all_current
        
    def _get_all_current(self):
        current_weather = self.weather_provider.fetch_current_weather(hours_to_fetch=self.past_hours_to_fetch)
        recent_level = self.level_provider.fetch_recent_level(hours_to_fetch=self.num_recent_samples)
    
        self._all_current = (current_weather, recent_level)

    def update_for_inference(self):
        self._get_all_current()

    @property
    def all_historical(self):
        if self._all_historical is None:
            self._get_all_historical()

        return self._all_historical 

    def _get_all_historical(self):
        historical_weather = self.weather_provider.fetch_historical_weather(self.weather_locs, self.name) 
        historical_level = self.level_provider.fetch_historical_level(self.usgs_gauge_id)
        self._all_historical = (historical_weather, historical_level)
