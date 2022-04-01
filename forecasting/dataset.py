from sklearn.preprocessing import MinMaxScaler
from darts import timeseries 
from darts.dataprocessing.transformers import Scaler
from darts import TimeSeries
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

        self.merged_X_train = self._merge(self.X_trains)
        self.merged_X_test = self._merge(self.X_tests)
        self.merged_X_validation = self._merge(self.X_validations)
        self.merged_X_current = self._merge(self.Xs_current)
  
    def _merge(self, Xs):
        frames = []
        for index, X_cur in enumerate(Xs):
            # Convert to df and add index as column name suffix
            cur_df = X_cur.pd_dataframe().rename(columns=lambda col: col+'_'+str(index))
            frames.append(cur_df)

        df = pd.concat(frames, axis=1)
        ts = TimeSeries.from_dataframe(df)

        return ts

        
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
             
            if X_cur.start_time() < y.start_time():
                _, X_cur = X_cur.split_before(y.start_time())   
            if X_cur.end_time() > y.end_time():
                X_cur, _ = X_cur.split_after(y.end_time()) 

            processed_Xs.append(X_cur)     

        if y.start_time() < processed_Xs[0].start_time():
            _, y = y.split_before(processed_Xs[0].start_time()) # y starts before X, drop y before X start
        if y.end_time() > processed_Xs[0].end_time(): # y ends after X, drop y after X end
            y, _ = y.split_after(processed_Xs[0].end_time())

        y = self.target_scaler.transform(y)


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
        self.catchment_data.update_for_inference()
        current_dfs = self.catchment_data.all_current_data.copy()
        self.Xs_current, self.y_current = self._pre_process(current_dfs, fit_scalers=False)
