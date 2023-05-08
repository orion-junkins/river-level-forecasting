import os
from typing import List, Optional, Sequence, Tuple

from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
import pandas as pd
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
    ) -> Tuple[TimeSeries, TimeSeries, TimeSeries, TimeSeries, TimeSeries, TimeSeries]:
        """
        Partition data using the specified test size. Splits into train, validation and test sets.

        Args:
            validation_size (int): Size of validation set in hours.
            test_size (int): Size of test set in hours.

        Returns:
            tuple[TimeSeries, TimeSeries, TimeSeries, TimeSeries, TimeSeries, TimeSeries]: (X_train, X_validation, X_test, y_train, y_validation, y_test)
        """
        train_validation_dividing_index = len(self.X) - (validation_size + test_size)
        validation_test_dividing_index = len(self.X) - (test_size)

        X_train = self.X[:train_validation_dividing_index]
        y_train = self.y[:train_validation_dividing_index]

        X_validation = self.X[train_validation_dividing_index: validation_test_dividing_index]
        y_validation = self.y[train_validation_dividing_index: validation_test_dividing_index]

        X_test = self.X[validation_test_dividing_index:]
        y_test = self.y[validation_test_dividing_index:]

        X_train = self.scaler.fit_transform(X_train)
        y_train = self.target_scaler.fit_transform(y_train)

        X_validation = self.scaler.transform(X_validation)
        y_validation = self.target_scaler.transform(y_validation)

        X_test = self.scaler.transform(X_test)
        y_test = self.target_scaler.transform(y_test)

        return X_train, X_validation, X_test, y_train, y_validation, y_test

class PartitionedTrainingDataset(TrainingDataset):
    """Dataset abstraction that fetches, processes and exposes needed X and y datasets given a CatchmentData instance."""

    def __init__(
        self,
        catchment_data: CatchmentData,
        cache_path: str,
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
        self.feature_partitions = [coordinate for coordinate in catchment_data.weather_provider.coordinates]
        self._cache_path = cache_path

        if not os.path.exists(self._cache_path):
            os.makedirs(self._cache_path)

        super().__init__(
            catchment_data,
            validation_size=validation_size,
            test_size=test_size,
            rolling_sum_columns=rolling_sum_columns,
            rolling_mean_columns=rolling_mean_columns,
            rolling_window_sizes=rolling_window_sizes
        )

    def load_feature_partition(self, partition_index: int) -> None:
        partition_path = self.generate_partition_path(partition_index)
        self.X = TimeSeries.from_dataframe(pd.read_parquet(partition_path))

        train_partition_path = self.generate_partition_path(partition_index, "train")
        validation_partition_path = self.generate_partition_path(partition_index, "validation")
        test_partition_path = self.generate_partition_path(partition_index, "test")

        if os.path.exists(train_partition_path):
            self.X_train = TimeSeries.from_dataframe(pd.read_parquet(train_partition_path))
        else:
            self.X_train = None

        if os.path.exists(validation_partition_path):
            self.X_validation = TimeSeries.from_dataframe(pd.read_parquet(validation_partition_path))
        else:
            self.X_validation = None

        if os.path.exists(test_partition_path):
            self.X_test = TimeSeries.from_dataframe(pd.read_parquet(test_partition_path))
        else:
            self.X_test = None


    def generate_partition_path(self, partition_index: int, subset_name: str = "") -> str:
        partition_coord = self.feature_partitions[partition_index]
        return os.path.join(self._cache_path, f"{self.prefix_for_lon_lat(partition_coord.lon, partition_coord.lat)}{'_' if subset_name else ''}{subset_name}.parquet")

    def _load_data(self) -> Tuple[TimeSeries, TimeSeries]:
        """Load and process data.

        Returns:
            tuple[TimeSeries, TimeSeries]: Tuple of (X and y) historical data.
        """
        # FIXME add functionality so that I don't have to skip over the CatchmentData object
        weather_provider = self.catchment_data.weather_provider
        level_provider = self.catchment_data.level_provider

        columns = self.catchment_data.columns

        historical_level = level_provider.fetch_historical_level()
        earliest_historical_level = historical_level.index.to_series().min().strftime("%Y-%m-%d")

        for coord in self.feature_partitions:
            weather_provider.coordinates = [coord]
            historical_weather = weather_provider.fetch_historical(start_date=earliest_historical_level, columns=columns)

            # this should not be that inefficient since not much preprocessing happens to historical_level
            X, y = self._pre_process(historical_weather, historical_level)

            partition_path = os.path.join(self._cache_path, f"{self.prefix_for_lon_lat(coord.lon, coord.lat)}.parquet")
            X.pd_dataframe(copy=False).to_parquet(partition_path)

        return X, y

    @property
    def num_feature_partitions(self) -> int:
        return len(self.feature_partitions)

    def prefix_for_lon_lat(self, lon: float, lat: float) -> str:
        return f"{lon:.2f}_{lat:.2f}"

    def _partition(
        self,
        validation_size: int,
        test_size: int
    ) -> Tuple[TimeSeries, TimeSeries, TimeSeries, TimeSeries, TimeSeries, TimeSeries]:
        """
        Partition data using the specified test size. Splits into train, validation and test sets.

        Args:
            validation_size (int): Size of validation set in hours.
            test_size (int): Size of test set in hours.

        Returns:
            tuple[TimeSeries, TimeSeries, TimeSeries, TimeSeries, TimeSeries, TimeSeries]: (X_train, X_validation, X_test, y_train, y_validation, y_test)
        """
        train_validation_dividing_index = len(self.y) - (validation_size + test_size)
        validation_test_dividing_index = len(self.X) - (test_size)

        y_train = self.y[:train_validation_dividing_index]
        y_validation = self.y[train_validation_dividing_index: validation_test_dividing_index]
        y_test = self.y[validation_test_dividing_index:]

        X_train_min_maxs = {}

        for i in range(self.num_feature_partitions):
            self.load_feature_partition(i)

            X = self.X
            X_train = X[:train_validation_dividing_index]
            X_validation = X[train_validation_dividing_index: validation_test_dividing_index]
            X_test = X[validation_test_dividing_index:]

            for col in X_train.columns:
                min_val = X_train[col].values().min()
                max_val = X_train[col].values().max()
                X_train_min_maxs[col] = [min_val, max_val]

            X_train = self.scaler.fit_transform(X_train)
            X_test = self.scaler.transform(X_test)
            X = self.scaler.transform(X)

            partition_path = self.generate_partition_path(i)
            validation_partition_path = self.generate_partition_path(i, "validation")
            train_partition_path = self.generate_partition_path(i, "train")
            test_partition_path = self.generate_partition_path(i, "test")

            X.pd_dataframe(copy=False).to_parquet(partition_path)
            X_train.pd_dataframe(copy=False).to_parquet(train_partition_path)
            X_validation.pd_dataframe(copy=False).to_parquet(validation_partition_path)
            X_test.pd_dataframe(copy=False).to_parquet(test_partition_path)

        # build the real scaler
        df = TimeSeries.from_dataframe(pd.DataFrame(X_train_min_maxs))
        self.scaler.fit(df)

        self.X = None

        return None, None, None, y_train, y_validation, y_test
