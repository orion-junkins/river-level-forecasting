from darts.dataprocessing.transformers import Scaler
from general_utilities.dataset_utilities import pre_process
from sklearn.preprocessing import MinMaxScaler


class TrainingDataset:
    """
    Dataset abstraction that fetches, processes and exposes needed X and y datasets given a CatchmentData instance.
    """
    def __init__(self, catchment_data, test_size=0.1, validation_size=0.2) -> None:
        """
        Generate a Dataset from a CatchmentData instance using the given test and validation sizes.

        Note that variable plurality indicates the presence of multiple datasets. Ie Xs_train is a list of multiple X_train sets.
        Args:
            catchment_data (CatchmentData): All needed catchment data.
            test_size (float, optional): Desired test set size. Defaults to 0.1.
            validation_size (float, optional): Desired validation size. Defaults to 0.2.
        """
        self.catchment_data = catchment_data
        self.scaler = Scaler(MinMaxScaler())
        self.target_scaler = Scaler(MinMaxScaler())

        self.Xs_historical, self.y_historical = self.load_data()
        self.Xs_train, self.Xs_test, self.Xs_validation, self.y_train, self.y_test, self.y_validation = self._partition(test_size, validation_size)
        # TODO add validation call - ie all X sets are same size, match y sets.

    def load_data(self):
        historical_weather, historical_level = self.catchment_data.all_historical_data
        Xs_historical, y_historical = pre_process(historical_weather, historical_level)
        return (Xs_historical, y_historical)   

    def _partition(self, test_size, validation_size):
        """
        Partition data using the specified test and validation sizes. Note that validation size is relative to whatever data is excluded from the test set.

        Args:
            test_size (float): Size of test set relative to overall dataset size.
            validation_size (float): Size of validation set relative to dataset size after excluding test set.

        Returns:
            tuple of (list of TimeSeries, list of TimeSeries, list of TimeSeries, TimeSeries, TimeSeries, TimeSeries): (Xs_train, Xs_test, Xs_validation, y_train, y_test, y_validation)
        """
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

        Xs_train = self.scaler.fit_transform(Xs_train)
        y_train = self.target_scaler.fit_transform(y_train)

        Xs_validation = self.scaler.transform(Xs_validation)
        y_validation = self.scaler.transform(y_validation)

        Xs_test = self.scaler.transform(Xs_test)
        y_test = self.scaler.transform(y_test)

        return (Xs_train, Xs_test, Xs_validation, y_train, y_test, y_validation)
