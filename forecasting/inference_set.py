from forecasting.data_fetching_utilities.weather import *
from forecasting.data_fetching_utilities.level import *
from forecasting.forecast_site import ForecastSite
from datetime import timedelta

class InferenceSet:
    def __init__(self, forecast_site, x_shape, scaler) -> None:
        self.data = forecast_site.all_inference_data
        self.x_shape = x_shape
        self._scale(scaler)

    def _scale(self, scaler):
        for column in self.data.columns:
            # fit and transform the self.data
            self.data[[column]] = scaler.fit_transform(self.data[[column]])

    def data_for_window(self, start, end=None, window_size_hours=5):
        if type(start) != 'datetime.datetime':
            start = datetime.fromisoformat(start)
            
        if end is None:
            end = start + timedelta(hours=window_size_hours-1)

        if start not in self.data.index:
            print("ERROR: invalid start timestamp")
            return None

        if end not in self.data.index:
            print("ERROR: invalid end timestamp")
            return None

        data_in_range = self.data.loc[start:end, :]
        data_in_range = data_in_range.values.reshape(self.x_shape)
        return data_in_range
        

