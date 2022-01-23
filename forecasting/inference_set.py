from data_fetching_utilities.weather import *
from data_fetching_utilities.level import *

class InferenceSet:
    def __init__(self, forecast_site, historical_weather_paths) -> None:
        self.forecast_site = forecast_site
        self.level = get_historical_level("14162500", start='1990-01-01')