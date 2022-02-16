from pandas import DataFrame
from datetime import timedelta

from forecasting.data_fetching_utilities.weather import *
from forecasting.data_fetching_utilities.level import *
from forecasting.general_utilities.df_utils import *

class PredictionSet:

    def __init__(self, catchment_data, dataset) -> None:
        """
        Produce a processed inference set for the given forecast site. Adheres to given shape/scaler.
        Args:
            catchment_data (CatchmentData): Forecast site from which data should be fetched.
            dataset (Dataset): primary Dataset instance. Needed for scalers and shape info.
            scaler (MinMaxScaler): Scaler used on model training data.
        """
        self.catchment_data = catchment_data
        self.X_shape = dataset.input_shape
        self.scaler = dataset.scaler
        self.target_scaler = dataset.target_scaler

        self.Xs, self.y = self._pre_process() # X is list of dfs, y is df
        

    def _pre_process(self):
        """
        Fetch needed data and perform all standard preprocessing.
        """
        dfs = self.catchment_data.all_current_data.copy()
        Xs = []
        y = None
        for df in dfs:
            print("adding lag")
            df = add_lag(df)
            print("done adding lag")
            X_cur, y_cur = split_X_y(df)
            X_cur = scale(X_cur, self.scaler, fit_scalers=False)
            Xs.append(X_cur)
            y = y_cur
            assert(y == None or y == y_cur)

        y = scale(y, self.target_scaler, fit_scalers=False)

        return Xs,y

    def update(self):
        """
        Force an update for forecast site data, re-fetch and re-process. Call hourly at minimum when forecasting.
        """
        self.catchment_data.update_for_inference()
        self.Xs, self.y = self._pre_process()