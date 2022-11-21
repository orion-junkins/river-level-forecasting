from typing import Optional

from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler

from rlf.forecasting.base_dataset import BaseDataset
from rlf.forecasting.catchment_data import CatchmentData


class InferenceDataset(BaseDataset):
    """Dataset abstraction that fetches, processes and exposes needed X and y datasets for inference given a CatchmentData instance."""

    def __init__(
        self,
        scaler: Scaler,
        target_scaler: Scaler,
        catchment_data: CatchmentData,
        rolling_sum_columns: Optional[list[str]] = None,
        rolling_mean_columns: Optional[list[str]] = None,
        rolling_window_sizes: list[int] = (10*24, 30*24)
    ) -> None:
        """
        Generate an inference Dataset from a CatchmentData instance using the given test and validation sizes.

        Note that variable plurality indicates the presence of multiple datasets. Ie Xs_train is a list of multiple X_train sets.

        Args:
            scaler (Scaler): Previously fit scaler for Xs from training forecaster.
            target_scaler (Scaler): Previously fit target scaler for y from training forecaster.
            catchment_data (CatchmentData): CatchmentData instance to use for inference.
            rolling_sum_columns (list[str], optional): Columns to generate rolling sums for. Defaults to None.
            rolling_mean_columns (list[str], optional): Columns to generate rolling means for. Defaults to None.
            rolling_window_sizes (list[int], optional): Different window sizes to use for rolling sums and means. If columns are specified for sums or means then a column for each window size will be generated. Defaults to 10 days (10 days * 24 hrs/day) and 30 days (30 days * 24 hrs/day).
        """
        super().__init__(
            catchment_data,
            rolling_sum_columns=rolling_sum_columns,
            rolling_mean_columns=rolling_mean_columns,
            rolling_window_sizes=rolling_window_sizes
        )
        self.scaler = scaler
        self.target_scaler = target_scaler

        self.Xs, self.y = self._get_data()

        # TODO add validation call - ie all X sets are same size, match y sets.

    def _get_data(self, update: bool = False) -> tuple[list[TimeSeries], TimeSeries]:
        """Retrieve data from catchment data instance. Update (re-fetch latest) only if specified.

        Args:
            update (bool, optional): Whether or not to refetch latest data. Defaults to False.

        Returns:
            tuple[list[TimeSeries], TimeSeries]: Tuple of (Xs, y) representing all current data.
        """
        # Update if specified
        if update:
            self.catchment_data.update_for_inference()

        # Fetch
        current_weather, recent_level = self.catchment_data.all_current

        # Process
        Xs_current, y_current = self._pre_process(current_weather, recent_level, allow_future_X=True)

        # Scale
        Xs_current = self.scaler.transform(Xs_current)
        y_current = self.target_scaler.transform(y_current)

        return (Xs_current, y_current)

    def update(self) -> None:
        """Update the underlying catchment data for inference with up to date data. This triggers new weather data queries and a rebuild of all current datasets."""
        self.Xs_current, self.y_current = self._get_data(update=True)
