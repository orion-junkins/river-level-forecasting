from forecasting.data_fetching_utilities.weather import *
from forecasting.data_fetching_utilities.level import *
from forecasting.forecast_site import ForecastSite
from datetime import timedelta

class InferenceSet:
    def __init__(self, forecast_site) -> None:
        self.data = forecast_site.all_inference_data

    def data_for_window(self, start, end=None, window_size_hours=5):
        if type(start) is not 'datetime.datetime':
            start = datetime.fromisoformat(start)
        print(type(start))
        if end is None:
            end = start + timedelta(hours=window_size_hours-1)

        if start not in self.data.index:
            print("ERROR: invalid start timestamp")
            return None

        if end not in self.data.index:
            print("ERROR: invalid end timestamp")
            return None

        data_in_range = self.data.loc[start:end, :]
        return data_in_range
        

