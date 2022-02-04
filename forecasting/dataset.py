from sklearn.preprocessing import MinMaxScaler

from forecasting.general_utilities.df_utils import *

class Dataset:

    def __init__(self, data_fetcher) -> None:
        self.scaler = MinMaxScaler()
        self.target_scaler = MinMaxScaler()
                
        self.X_historical_processed, self.y_historical_processed = self._pre_process(data_fetcher.all_historical_data.copy())
        self.X_historical_windowed, self.y_historical_windowed = get_all_windows(self.X_historical_processed, self.y_historical_processed)
        self.X_train, self.X_test, self.y_train, self.y_test = partition(self.X_historical_windowed, self.y_historical_windowed)

    def _pre_process(self, df):
        df = add_lag(df)
        X, y = split_X_y(df)
        X = scale(X, self.scaler)
        y = scale(y, self.target_scaler)
        return (X, y)


    @property
    def X_train_shaped(self):
        return self.X_train.reshape(self.X_train.shape[0], self.X_train.shape[1], self.X_train.shape[2], 1)
    @property
    def X_test_shaped(self):
        return self.X_test.reshape(self.X_test.shape[0], self.X_test.shape[1], self.X_test.shape[2], 1)

    @property
    def input_shape(self):
        print(self.X_train_shaped[0].shape)
        return self.X_train_shaped[0].shape