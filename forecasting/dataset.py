from sklearn.preprocessing import MinMaxScaler

from forecasting.general_utilities.df_utils import *

class Dataset:

    def __init__(self, catchment_data, prediction_only=False) -> None:
        self.catchment_data = catchment_data
        self.scaler = MinMaxScaler()
        self.target_scaler = MinMaxScaler()

        historical_dfs = self.catchment_data.all_historical_data.copy()
        self.Xs_historical, self.y_historical = self._pre_process(historical_dfs)

        current_dfs = self.catchment_data.all_current_data.copy()
        self.Xs_current, self.y_current = self._pre_process(current_dfs, fit_X_scaler=False, fit_y_scaler=False)


    def _pre_process(self, dfs, fit_X_scaler=True, fit_y_scaler=True):
        """
        Fetch needed data and perform all standard preprocessing.
        """
        Xs = []
        y = None
        for df in dfs:
            print("adding lag")
            df = add_lag(df)
            print("done adding lag")
            X_cur, y_cur = split_X_y(df)
            X_cur = scale(X_cur, self.scaler, fit_scalers=fit_X_scaler)
            fit_X_scaler = False # Only fit the scaler on the first iteration
            Xs.append(X_cur)

            #assert(y == None or y == y_cur)
            y = y_cur

        y = scale(y, self.target_scaler, fit_scalers=fit_y_scaler)

        return Xs,y
