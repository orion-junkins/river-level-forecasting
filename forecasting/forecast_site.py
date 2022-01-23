from forecasting.time_utils import date_days_ago
from forecasting.data_fetching_utilities.weather import get_all_historical_weather
from forecasting.data_fetching_utilities.level import get_historical_level

class ForecastSite:
    """
    Basic data wrapper class to track individual forecast sites
    """
    def __init__(self, gauge_id, weather_paths) -> None:
        self.gauge_id = gauge_id
        self.weather_paths = weather_paths

        # Level dataframe (indexed by 'datetime', columns = ['level'])
        self.hisotrical_level = get_historical_level(gauge_id)
        self.recent_level = get_historical_level(start=date_days_ago(5))\

        # List of weather dataframes (indexed by 'datetime', columns = ['temp', 'rain', etc.])
        self.historical_weather = get_all_historical_weather(self.weather_paths)
    

    def update_for_inference(self):
        self.recent_level = get_historical_level(start=date_days_ago(5))


