from sklearn.preprocessing import MinMaxScaler
from darts import timeseries 
from darts.dataprocessing.transformers import Scaler

class Dataset:
    def __init__(self, catchment_data, test_size=0.1, validation_size=0.2) -> None:
        self.catchment_data = catchment_data
        self.scaler = Scaler(MinMaxScaler())
        self.target_scaler = Scaler(MinMaxScaler())

        historical_weather, historical_level = self.catchment_data.all_historical_data
        self.Xs_historical, self.y_historical = self._pre_process(historical_weather, historical_level)
        self.Xs_train, self.Xs_test, self.Xs_validation, self.y_train, self.y_test, self.y_validation = self._partition(test_size, validation_size)

        current_weather, recent_level = self.catchment_data.all_current_data
        self.Xs_current, self.y_current = self._pre_process(current_weather, recent_level, allow_future_X=True)
        self.Xs_current, self.y_current = self.scale_data(self.Xs_current, self.y_current)

    
    @property
    def num_training_samples(self):
        return len(self.Xs_train[0])


    @property
    def num_X_sets(self):
        return len(self.Xs_train)

        
    def _pre_process(self, Xs, y, allow_future_X=False):
        """
        Fetch needed data and perform all standard preprocessing.
        """
        y = timeseries.TimeSeries.from_dataframe(y)

        processed_Xs = []

        for X_cur in Xs:
            X_cur = self.add_engineered_features(X_cur)
            X_cur = timeseries.TimeSeries.from_dataframe(X_cur)

            if X_cur.start_time() < y.start_time():
                _, X_cur = X_cur.split_before(y.start_time())    # X starts before y, drop X before y start

            if not allow_future_X:
                if X_cur.end_time() > y.end_time():
                    X_cur, _ = X_cur.split_after(y.end_time())  # X ends after y, drop X after y end

            processed_Xs.append(X_cur)     

        if y.start_time() < processed_Xs[0].start_time():
            _, y = y.split_before(processed_Xs[0].start_time()) # y starts before X, drop y before X start

        if y.end_time() > processed_Xs[0].end_time(): # y ends after X, drop y after X end
            y, _ = y.split_after(processed_Xs[0].end_time())

        return (processed_Xs, y)


    def scale_data(self, Xs, y, fit_scalers=False):
        if fit_scalers:
            self.scaler = self.scaler.fit(Xs)
            self.target_scaler = self.target_scaler.fit(y)
        Xs = self.scaler.transform(Xs)
        y = self.target_scaler.transform(y)
        return (Xs, y)
        

    def _partition(self, test_size, validation_size):
        Xs_train = []
        Xs_test = []
        Xs_validation = []
        for X in self.Xs_historical:
            X_train, X_test = X.split_after(1-test_size)
            X_train, X_validation = X_train.split_after(1-validation_size)

            Xs_train.append(X_train)
            Xs_test.append(X_test)

            Xs_validation.append(X_validation)

        y_train, y_test = self.y_historical.split_after(1-test_size)
        y_train, y_validation = y_train.split_after(1-validation_size)

        Xs_train, y_train = self.scale_data(Xs_train, y_train, fit_scalers=True)
        Xs_validation, y_validation = self.scale_data(Xs_validation, y_validation)
        Xs_test, y_test = self.scale_data(Xs_test, y_test)

        return (Xs_train, Xs_test, Xs_validation, y_train, y_test, y_validation)
    

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
        current_weather, recent_level = self.catchment_data.all_current_data
        self.Xs_current, self.y_current = self._pre_process(current_weather, recent_level, allow_future_X=True)
        self.Xs_current, self.y_current = self.scale_data(self.Xs_current, self.y_current)