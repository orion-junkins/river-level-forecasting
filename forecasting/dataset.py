from sklearn.preprocessing import MinMaxScaler
from darts import timeseries 
from darts.dataprocessing.transformers import Scaler
import pandas as pd

class Dataset:

    def __init__(self, catchment_data) -> None:
        self.catchment_data = catchment_data
        self.scaler = Scaler(MinMaxScaler())
        self.target_scaler = Scaler(MinMaxScaler())

        historical_weather, historical_level = self.catchment_data.all_historical_data
        self.Xs_historical, self.y_historical = self._pre_process(historical_weather, historical_level)
        self.X_trains, self.X_tests, self.X_validations, self.y_train, self.y_test, self.y_validation = self._partition()

        current_weather, recent_level = self.catchment_data.all_current_data
        self.Xs_current, self.y_current = self._pre_process(current_weather, recent_level, fit_scalers=False)
        
  
        
    @property
    def num_X_sets(self):
        return len(self.X_trains)

        
    def _pre_process(self, Xs, y, fit_scalers=True):
        """
        Fetch needed data and perform all standard preprocessing.
        """
        y = timeseries.TimeSeries.from_dataframe(y)
        if fit_scalers:
            self.target_scaler.fit(y)

        processed_Xs = []

        for X_cur in Xs:
            X_cur = self.add_engineered_features(X_cur)
            X_cur = timeseries.TimeSeries.from_dataframe(X_cur)
            if fit_scalers:
                self.scaler.fit(X_cur)
                fit_scalers = False # Only fit scalers once
            X_cur = self.scaler.transform(X_cur)
            processed_Xs.append(X_cur)        

        if y.start_time() < processed_Xs[0].start_time():
            y = y.drop_before(processed_Xs[0].start_time())

        y = self.target_scaler.transform(y)

        print(processed_Xs[0].start_time())
        print(y.start_time())

        return (processed_Xs, y)

    def _partition(self, test_size=0.2, validation_size=0.2):
        X_trains = []
        X_tests = []
        X_validations = []
        for X in self.Xs_historical:
            X_train, X_test = X.split_after(1-test_size)
            X_train, X_validation = X_train.split_after(1-validation_size)

            X_trains.append(X_train)
            X_tests.append(X_test)
            X_validations.append(X_validation)

        y_train, y_test = self.y_historical.split_after(1-test_size)
        y_train, y_validation = y_train.split_after(1-validation_size)

        return (X_trains, X_tests, X_validations, y_train, y_test, y_validation)
    
    def add_engineered_features(self, df):
        df['day_of_year'] = df.index.day_of_year

        df['snow_10d'] = df['snow_1h'].rolling(window=10 * 24).sum()
        df['snow_30d'] = df['snow_1h'].rolling(window=30 * 24).sum()
        df['rain_10d'] = df['rain_1h'].rolling(window=10 * 24).sum()
        df['rain_30d'] = df['rain_1h'].rolling(window=30 * 24).sum()

        df['temp_10d'] = df['temp'].rolling(window=10 * 24).mean()
        df['temp_30d'] = df['temp'].rolling(window=30 * 24).mean()
        df.dropna(inplace=True)
        return df

    def update(self):
        self.Xs_current, self.y_current = self._pre_process(current_dfs, fit_scalers=False)