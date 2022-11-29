from typing import Optional

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
        test_size: float = 0.1,
        rolling_sum_columns: Optional[list[str]] = None,
        rolling_mean_columns: Optional[list[str]] = None,
        rolling_window_sizes: list[int] = (10*24, 30*24)
    ) -> None:
        """Generate a Dataset for training from a CatchmentData instance.

        Note that variable plurality indicates the presence of multiple datasets. E.g. Xs_train is a list of multiple X_train sets.

        Args:
            catchment_data (CatchmentData): CatchmentData instance to use for training.
            test_size (float, optional): Desired test set size. Defaults to 0.1.
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
        self.X_train, self.X_test, self.y_train, self.y_test = self._partition(test_size)
        # TODO add validation call - ie all X sets are same size, match y sets.

    def _load_data(self) -> tuple[list[TimeSeries], TimeSeries]:
        """Load and process data.

        Returns:
            tuple[list[TimeSeries], TimeSeries]: Tuple of (Xs and y) historical data.
        """
        historical_weather, historical_level = self.catchment_data.all_historical
        X, y = self._pre_process(historical_weather, historical_level)
        return X, y

    def _partition(
        self,
        test_size: float
    ) -> tuple[list[TimeSeries], list[TimeSeries], TimeSeries, TimeSeries]:
        """
        Partition data using the specified test size.

        Args:
            test_size (float): Size of test set relative to overall dataset size.

        Returns:
            tuple[list[TimeSeries], list[TimeSeries], TimeSeries, TimeSeries]: (Xs_train, Xs_test, y_train, y_test)
        """
        X_train, X_test = self.X.split_after(1-test_size)

        y_train, y_test = self.y.split_after(1-test_size)

        X_train = self.scaler.fit_transform(X_train)
        y_train = self.target_scaler.fit_transform(y_train)

        X_test = self.scaler.transform(X_test)
        y_test = self.target_scaler.transform(y_test)

        return X_train, X_test, y_train, y_test
