from data_fetching_utilities.weather import *
from data_fetching_utilities.level import *

class InferenceSet:
    def __init__(self, locations, level_url) -> None:
        self.forecasts = fetch_forecasts(locations)
        self.level = get_historical_level("14162500", start='1990-01-01')