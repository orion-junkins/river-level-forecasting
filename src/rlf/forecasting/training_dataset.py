from typing import List, Optional, Sequence, Tuple

from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from sklearn.preprocessing import MinMaxScaler

from rlf.forecasting.base_dataset import BaseDataset
from rlf.forecasting.catchment_data import CatchmentData


class TrainingDataset(BaseDataset):
    """Dataset abstraction that fetches, processes and exposes needed X and y datasets given a CatchmentData instance."""

    def __init__(
        self,
        catchment_data: CatchmentData,
        validation_size: int = 24 * 365 * 3,
        test_size: int = 24 * 365 * 3,
        rolling_sum_columns: Optional[List[str]] = None,
        rolling_mean_columns: Optional[List[str]] = None,
        rolling_window_sizes: Sequence[int] = (10 * 24, 30 * 24)
    ) -> None:
        """Generate a Dataset for training from a CatchmentData instance.

        Note that variable plurality indicates the presence of multiple datasets. E.g. Xs_train is a list of multiple X_train sets.

        Args:
            catchment_data (CatchmentData): CatchmentData instance to use for training.
            validation_size (int, optional): Size of validation set in hours. Defaults to 3 years (365 days * 24 hours/day * 3 years).
            test_size (int, optional): Size of test set in hours. Defaults to 3 years (365 days * 24 hours/day * 3 years).
            rolling_sum_columns (list[str], optional): List of columns to compute rolling sums for. Defaults to None.
            rolling_mean_columns (list[str], optional): List of columns to compute rolling means for. Defaults to None.
            rolling_window_sizes (list[int], optional): Window sizes to use for rolling computations. Defaults to 10 days (10 days * 24 hrs/day) and 30 days (30 days * 24 hrs/day).
        """
        super().__init__(
            catchment_data,
            rolling_sum_columns=rolling_sum_columns,
            rolling_mean_columns=rolling_mean_columns,
            rolling_window_sizes=rolling_window_sizes
        )
        self.scaler = Scaler(MinMaxScaler())
        self.target_scaler = Scaler(MinMaxScaler())
        self.X, self.y = self._load_data()
        if len(self.X) <= test_size + validation_size:
            raise ValueError(f"The sum of test size ({test_size}) and validation size ({validation_size}) must be less than the total number of samples ({len(self.X)}).")

        self.X_train, self.X_validation, self.X_test, self.y_train, self.y_validation, self.y_test = self._partition(validation_size=validation_size, test_size=test_size)

    def _load_data(self) -> Tuple[TimeSeries, TimeSeries]:
        """Load and process data.

        Returns:
            tuple[TimeSeries, TimeSeries]: Tuple of (X and y) historical data.
        """
        historical_weather, historical_level = self.catchment_data.all_historical
        X, y = self._pre_process(historical_weather, historical_level)
        return X, y

    def _partition(
        self,
        validation_size: int,
        test_size: int
    ) -> Tuple[List[TimeSeries], List[TimeSeries], List[TimeSeries], TimeSeries, TimeSeries, TimeSeries]:
        """
        Partition data using the specified test size. Splits into train, validation and test sets.

        Args:
            validation_size (int): Size of validation set in hours.
            test_size (int): Size of test set in hours.

        Returns:
            tuple[List[TimeSeries], List[TimeSeries], List[TimeSeries], TimeSeries, TimeSeries, TimeSeries]: (X_train, X_validation, X_test, y_train, y_validation, y_test)
        """
        train_valididation_dividing_index = len(self.X) - (validation_size + test_size)
        valididation_test_dividing_index = len(self.X) - (test_size)

        X_train = self.X[:train_valididation_dividing_index]
        y_train = self.y[:train_valididation_dividing_index]

        X_validation = self.X[train_valididation_dividing_index: valididation_test_dividing_index]
        y_validation = self.y[train_valididation_dividing_index: valididation_test_dividing_index]

        X_test = self.X[valididation_test_dividing_index:]
        y_test = self.y[valididation_test_dividing_index:]

        X_train = self.scaler.fit_transform(X_train)
        y_train = self.target_scaler.fit_transform(y_train)

        X_validation = self.scaler.transform(X_validation)
        y_validation = self.target_scaler.transform(y_validation)

        X_test = self.scaler.transform(X_test)
        y_test = self.target_scaler.transform(y_test)

        return X_train, X_validation, X_test, y_train, y_validation, y_test
