from darts.dataprocessing.transformers import Scaler
from sklearn.preprocessing import MinMaxScaler

from rlf.forecasting.general_utilities.dataset_utilities import pre_process


class TrainingDataset:
    """
    Dataset abstraction that fetches, processes and exposes needed X and y datasets given a CatchmentData instance.
    """
    def __init__(self, catchment_data, test_size=0.1, rolling_sum_columns=[], rolling_mean_columns=[], rolling_window_sizes=[10*24, 30*24]) -> None:
        """Generate a Dataset for training from a CatchmentData instance.

        Note that variable plurality indicates the presence of multiple datasets. Ie Xs_train is a list of multiple X_train sets.


        Args:
            catchment_data (CatchmentData): All needed catchment data.
            test_size (float, optional): Desired test set size. Defaults to 0.1.
            rolling_sum_columns (list[str], optional): For which columns should a rolling sum variable be engineered. Defaults to [].
            rolling_mean_columns (list[str], optional): For which columns should a rolling mean variable be engineered. Defaults to [].
            rolling_window_sizes (list[int], optional): For which window sizes should rolling sum and mean variables be engineered. Defaults to [10*24, 30*24] (10 days and 30 days).
        """
        self.catchment_data = catchment_data
        self.scaler = Scaler(MinMaxScaler())
        self.target_scaler = Scaler(MinMaxScaler())
        self.rolling_sum_columns = rolling_sum_columns
        self.rolling_mean_columns = rolling_mean_columns
        self.rolling_window_sizes = rolling_window_sizes

        self.Xs_historical, self.y_historical = self._load_data()
        self.Xs_train, self.Xs_test, self.y_train, self.y_test = self._partition(test_size)
        # TODO add validation call - ie all X sets are same size, match y sets.

    def _load_data(self):
        historical_weather, historical_level = self.catchment_data.all_historical
        Xs_historical, y_historical = pre_process(historical_weather, historical_level, rolling_sum_columns=self.rolling_sum_columns, rolling_mean_columns=self.rolling_mean_columns, window_sizes=self.rolling_window_sizes)
        return (Xs_historical, y_historical)

    def _partition(self, test_size):
        """
        Partition data using the specified test size. 
        Args:
            test_size (float): Size of test set relative to overall dataset size.

        Returns:
            tuple of (list of TimeSeries, list of TimeSeries, list of TimeSeries, TimeSeries, TimeSeries, TimeSeries): (Xs_train, Xs_test, y_train, y_test)
        """
        Xs_train = []
        Xs_test = []
        for X in self.Xs_historical:
            X_train, X_test = X.split_after(1-test_size)

            Xs_train.append(X_train)
            Xs_test.append(X_test)

        y_train, y_test = self.y_historical.split_after(1-test_size)

        Xs_train = self.scaler.fit_transform(Xs_train)
        y_train = self.target_scaler.fit_transform(y_train)

        Xs_test = self.scaler.transform(Xs_test)
        y_test = self.scaler.transform(y_test)

        return (Xs_train, Xs_test, y_train, y_test)
