from pandas import DataFrame
from forecasting.data_fetching_utilities.weather import *
from forecasting.data_fetching_utilities.level import *
from datetime import timedelta

class PredictionSet:

    def __init__(self, forecast_site, x_shape, scaler) -> None:
        """
        Produce a processed inference set for the given forecast site. Adheres to given shape/scaler.
        Args:
            forecast_site (ForecastSite): Forecast site from which data should be fetched.
            x_shape (tuple): input shape for target model.
            scaler (MinMaxScaler): Scaler used on model training data.
        """
        self.forecast_site = forecast_site
        self.x_shape = x_shape
        self.scaler = scaler

        self.df = self._pre_process()
        

    def _pre_process(self) -> DataFrame:
        """
        Fetch needed data and perform all standard preprocessing.
        Returns:
            df (DataFrame): Processed dataframe.
        """
        df = self.forecast_site.all_inference_data
        for column in df.columns:
            # fit and transform the current column
            df[[column]] = self.scaler.fit_transform(df[[column]])
        return df


    def x_in_for_window(self, start, end=None, window_size_hours=5) -> DataFrame:
        """
        Grab an isolated dataframe for the given window.
        Args:
            start (datetime): Start of window.
            end (datetime, optional): End of window. Alternative to window_size_hours Defaults to None.
            window_size_hours (int, optional): Duration of window in hours. Alternative to end. Defaults to 5.
        Returns:
            x_in (np.array): dataset trimmed down to the given window. Reshaped according to self.x_shape.
        """
        # Typechecks and guard clauses to increase callsite flexibility TODO: minimize need upstream
        if type(start) != 'datetime.datetime':
            start = datetime.fromisoformat(start)
        if type(end) != 'datetime.datetime':
            start = datetime.fromisoformat(start)  
        if end is None:
            end = start + timedelta(hours=window_size_hours-1)
        if start not in self.df.index:
            print("ERROR: invalid start timestamp")
            return None
        if end not in self.df.index:
            print("ERROR: invalid end timestamp")
            return None

        x_in = self.df.loc[start:end, :]
        x_in = x_in.values.reshape(self.x_shape)
        return x_in
    
    def update(self):
        """
        Force an update for forecast site data, re-fetch and re-process. Call hourly at minimum when forecasting.
        """
        self.forecast_site.update_for_inference()
        self.df = self._pre_process()