from data_fetching_utilities.open_weather import *
from data_fetching_utilities.nwis import *

class InferenceSet:
    def __init__(self, locations, level_url) -> None:
        self.forecasts = fetch_forecasts(locations)
        self.level = get_hourly("14162500", start='1990-01-01')