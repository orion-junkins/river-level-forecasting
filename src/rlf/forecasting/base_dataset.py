from abc import ABC
from typing import Optional

from darts import TimeSeries
from darts.timeseries import concatenate
from pandas import DataFrame, Timestamp

from rlf.forecasting.catchment_data import CatchmentData
from rlf.forecasting.data_fetching_utilities.coordinate import Coordinate
from rlf.forecasting.data_fetching_utilities.weather_provider.weather_datum import WeatherDatum


class BaseDataset(ABC):
    """Abstract base class for all Datasets."""

    def __init__(
        self,
        catchment_data: CatchmentData,
        rolling_sum_columns: Optional[list[str]] = None,
        rolling_mean_columns: Optional[list[str]] = None,
        rolling_window_sizes: list[int] = (10*24, 30*24)
    ) -> None:
        """Create a new Dataset instance.

        Args:
            catchment_data (CatchmentData): All needed catchment data.
            rolling_sum_columns (list[str], optional): For which columns should a rolling sum variable be engineered. Defaults to [].
            rolling_mean_columns (list[str], optional): For which columns should a rolling mean variable be engineered. Defaults to [].
            rolling_window_sizes (list[int], optional): For which window sizes should rolling sum and mean variables be engineered. Defaults to [10*24, 30*24] (10 days and 30 days).
        """
        self.catchment_data = catchment_data
        self.rolling_sum_columns = rolling_sum_columns if rolling_sum_columns is not None else []
        self.rolling_mean_columns = rolling_mean_columns if rolling_mean_columns is not None else []
        self.rolling_window_sizes = rolling_window_sizes
        self.subsets = dict()

    def _pre_process(
        self,
        Xs: list[WeatherDatum],
        y: DataFrame,
        allow_future_X: bool = False
    ) -> tuple[list[TimeSeries], TimeSeries]:
        """
        Pre process data. This includes adding engineered features and trimming datasets to ensure X, y consistency.

        Args:
            Xs (list[WeatherDatum]): List of all X sets.
            y (DataFrame): Dataframe containing y set.
            allow_future_X (bool, optional): Determines if end date of X sets can exceed end date of y set. Only True for current data which includes forecasts (level data into future is not yet known, but weather is). Defaults to False.

        Returns:
            tuple[list[TimeSeries], TimeSeries]: Tuple containing (processed_Xs, y)
        """
        # find the boundaries that are valid for all datasets
        first_date, last_date = self._find_timestamp_boundaries(Xs, y)

        y = TimeSeries.from_dataframe(y).slice(first_date, last_date)

        X_last_date = None if allow_future_X else last_date
        processed_Xs = [self._process_datum(datum, first_date, X_last_date) for datum in Xs]
        global_X = concatenate(processed_Xs, axis="component")

        return global_X, y

    def _process_datum(self, datum: WeatherDatum, first_date: Timestamp, last_date: Timestamp | None) -> TimeSeries:
        """Process a single X datum.

        Processing an X datum involves cleaning the data, adding engineered features, renaming the columns, bounding the time index, and converting to a TimeSeries.
        NaNs that are not trailing will be linearly interpolated.
        The subsets attribute will be updated with the generated prefix for this datum.

        Args:
            datum (WeatherDatum): Datum to process.
            first_date (Timestamp): First allowed date for the time index. Any dates prior to this should be dropped.
            last_date (Timestamp): Last allowed date for the time index. Any dates after this should be dropped. If None then no bounding in this direction is done.

        Raises:
            ValueError: If the generated prefix for this datum already exists.

        Returns:
            TimeSeries: Processed datum.
        """
        X = datum.hourly_parameters

        X = self._strip_trailing_nans(X)

        X.interpolate(inplace=True)

        X = self._add_engineered_features(X)

        prefix = f"{datum.longitude:.3f}_{datum.latitude:.3f}_"
        if prefix not in self.subsets:
            X.columns = [prefix + c for c in X.columns]
            self.subsets[prefix] = Coordinate(lon=datum.longitude, lat=datum.latitude)
        else:
            raise ValueError(f"Prefix will be represented twice in the global X set: {prefix}")

        print(f"Length of X: {len(X)}")

        X = TimeSeries.from_dataframe(X)

        if last_date is not None:
            X = X.slice(first_date, last_date)
        else:
            _, X = X.split_before(first_date)

        return X

    @staticmethod
    def _strip_trailing_nans(df: DataFrame) -> DataFrame:
        """Strip out the trailing NaNs that can be found in WeatherProvider data.

        Trailing NaNs are found by detecting the latest date with a non-NaN value in any column and dropping everything after that.

        Args:
            df (DataFrame): DataFrame to remove trailing NaNs from.

        Returns:
            DataFrame: DataFrame with trailing NaNs removed.
        """
        last_date = df.apply(lambda x: x.last_valid_index()).max()
        df = df[df.index.to_series() <= last_date].copy()
        return df

    def _add_engineered_features(self, df: DataFrame) -> DataFrame:
        """
        Generate and add engineered features.

        Args:
            df (DataFrame): Data from which features should be engineered.

        Returns:
            DataFrame: Data including new features.
        """
        df['day_of_year'] = df.index.day_of_year

        for window_size in self.rolling_window_sizes:
            for rolling_sum_col in self.rolling_sum_columns:
                new_col_name = f"{rolling_sum_col}_sum_{window_size}"
                df[new_col_name] = df[rolling_sum_col].rolling(window=window_size).sum()

            for rolling_mean_col in self.rolling_mean_columns:
                new_col_name = f"{rolling_mean_col}_mean_{window_size}"
                df[new_col_name] = df[rolling_mean_col].rolling(window=window_size).mean()

        df.dropna(inplace=True)

        return df

    @staticmethod
    def _find_timestamp_boundaries(Xs: list[DataFrame], y: DataFrame) -> tuple[Timestamp, Timestamp]:
        """Find the first and last timestamp that guarantees data in all datasets.

        Args:
            Xs (list[DataFrame]): All X dataframes to check.
            y (DataFrame): y dataframe to check.

        Returns:
            tuple[Timestamp, Timestamp]: first timestamp, last timestamp inclusive.
        """
        all_dfs = [X.hourly_parameters for X in Xs] + [y]
        first_timestamp = max([df.index.to_series().min() for df in all_dfs])
        last_timestamp = min([df.apply(lambda x: x.last_valid_index()).max() for df in all_dfs])
        return first_timestamp, last_timestamp
