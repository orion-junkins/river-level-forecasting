from abc import ABC
from typing import Optional

from darts import TimeSeries
from pandas import DataFrame

from rlf.forecasting.catchment_data import CatchmentData
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
        y = TimeSeries.from_dataframe(y)

        processed_Xs = []

        for X_datum in Xs:
            X = X_datum.hourly_parameters

            X = self._add_engineered_features(X)

            prefix = f"{X_datum.longitude:.3f}_{X_datum.latitude:.3f}_"
            X.columns = [prefix + c for c in X.columns]

            X = TimeSeries.from_dataframe(X)

            if X.start_time() < y.start_time():
                _, X = X.split_before(y.start_time())    # X starts before y, drop X before y start

            if not allow_future_X:
                if X.end_time() > y.end_time():
                    X, _ = X.split_after(y.end_time())  # X ends after y, drop X after y end

            processed_Xs.append(X)

        if y.start_time() < processed_Xs[0].start_time():
            _, y = y.split_before(processed_Xs[0].start_time())  # y starts before X, drop y before X start

        if y.end_time() > processed_Xs[0].end_time():  # y ends after X, drop y after X end
            y, _ = y.split_after(processed_Xs[0].end_time())

        return processed_Xs, y

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
