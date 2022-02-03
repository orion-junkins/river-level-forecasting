import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

class Dataset:

    def __init__(self, forecast_site) -> None:
        print("starting")
        self.scaler = MinMaxScaler()
        self.target_scaler = MinMaxScaler()

        self.data = self._pre_process(forecast_site.all_training_data)
        self.X, self.y = self._build_windowed_X_y()
        self.X_train, self.X_test, self.y_train, self.y_test = self._partition()
        
        # TODO Investigate best location in pipeline for reshaping
        self.X_train_shaped = self.X_train.reshape(self.X_train.shape[0], self.X_train.shape[1], self.X_train.shape[2], 1)
        self.X_test_shaped = self.X_test.reshape(self.X_test.shape[0], self.X_test.shape[1], self.X_test.shape[2], 1)
        
        self.input_shape = self.X_train_shaped[0].shape
        
        assert(self.data.isna().sum().sum() == 0)


    def _pre_process(self, training_data):
        data = training_data.copy()
        data = self._shift_targets(data)
        data = self._scale(data)
        return data

    def _shift_targets(self, data):
        data['target_level'] = np.nan
        rows = data.shape[0]
        for row_idx in range(0, rows-1):
            data['target_level'][row_idx] = data['level'][row_idx+1]
        data.drop(data.tail(1).index,inplace=True)
        return data
    

    def _scale(self, data):
         # For every feature column,
        for column in data.columns[:-1]:
            # fit and transform the data
            data[[column]] = self.scaler.fit_transform(data[[column]])

        # Scale the target column
        target_col = data.columns[-1]
        data[[target_col]] = self.target_scaler.fit_transform(data[[target_col]])

        return data
    

    def _build_windowed_X_y(self, window_length=5):
        X = self.data.iloc[:,:-1].values
        y = self.data.iloc[:,-1].values

        num_samples = len(X) - window_length

        windowed_X = []
        windowed_y = []
        for index in range(num_samples):
            current_window_end = index + window_length
            cur_X_seq = X[index:current_window_end, :]
            windowed_X.append(cur_X_seq)

            windowed_y.append(y[current_window_end])

        X = np.array(windowed_X)
        y = np.array(windowed_y)
        
        return (X, y)

    def _partition(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size = 0.2, random_state = 0)
        return (X_train, X_test, y_train, y_test)
