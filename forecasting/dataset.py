from sklearn.preprocessing import MinMaxScaler
from darts import timeseries 
from forecasting.general_utilities.df_utils import *

class Dataset:

    def __init__(self, catchment_data) -> None:
        self.catchment_data = catchment_data
        self.scaler = MinMaxScaler()
        self.target_scaler = MinMaxScaler()

        historical_dfs = self.catchment_data.all_historical_data.copy()
        self.Xs_historical, self.y_historical = self._pre_process(historical_dfs)
        self.X_trains, self.X_tests, self.X_validations = self._partition()

        current_dfs = self.catchment_data.all_current_data.copy()
        self.Xs_current, self.y_current = self._pre_process(current_dfs, fit_X_scaler=False, fit_y_scaler=False)


    def _pre_process(self, dfs, fit_X_scaler=True, fit_y_scaler=True):
        """
        Fetch needed data and perform all standard preprocessing.
        """
        Xs = []
        y = None
        for df in dfs:
            X_cur, y_cur = split_X_y(df)
            X_cur = scale(X_cur, self.scaler, fit_scalers=fit_X_scaler)
            fit_X_scaler = False # Only fit the scaler on the first iteration
            X_cur = timeseries.TimeSeries.from_dataframe(X_cur)
            Xs.append(X_cur)

            #assert(y == None or y == y_cur)
            y = y_cur

        y = scale(y, self.target_scaler, fit_scalers=fit_y_scaler)
        y =  timeseries.TimeSeries.from_dataframe(y_cur)

        return Xs,y

    def _partition(self, test_size=0.1, validation_size=0.2):
        X_trains = []
        X_tests = []
        X_validations = []
        for X in self.Xs_historical:
            X_train, X_test = X.split_after(1-test_size)
            X_train, X_validation = X_train.split_after(1-validation_size)

            X_trains.append(X_train)
            X_tests.append(X_test)
            X_validations.append(X_validation)

        return (X_trains, X_tests, X_validations)