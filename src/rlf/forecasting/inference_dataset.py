from darts import timeseries
from darts.dataprocessing.transformers.invertible_data_transformer import InvertibleDataTransformer

from rlf.forecasting.base_dataset import BaseDataset


class InferenceDataset(BaseDataset):
    """Dataset abstraction that fetches, processes and exposes needed X and y datasets for inference given a CatchmentData instance."""

    def __init__(self, scaler: InvertibleDataTransformer, target_scaler: InvertibleDataTransformer, **kwargs) -> None:
        """
        Generate an inference Dataset from a CatchmentData instance using the given test and validation sizes.

        Note that variable plurality indicates the presence of multiple datasets. Ie Xs_train is a list of multiple X_train sets.

        Args:
            scaler (InvertibleDataTransformer): Previously fit scaler for Xs from training forecaster.
            target_scaler (InvertibleDataTransformer): Previously fit target scaler for y from training forecaster.
        """
        super().__init__(**kwargs)
        self.scaler = scaler
        self.target_scaler = target_scaler

        self.Xs, self.y = self._get_data()

        # TODO add validation call - ie all X sets are same size, match y sets.

    def _get_data(self, update: bool = False) -> tuple[list[timeseries], timeseries]:
        """Retrieve data from catchment data instance. Update (re-fetch latest) only if specified.

        Args:
            update (bool, optional): Whether or not to refetch latest data. Defaults to False.

        Returns:
            tuple[list[timeseries], timeseries]: Tuple of (Xs, y) representing all current data.
        """
        # Update if specified
        if update:
            self.catchment_data.update_for_inference()

        # Fetch
        current_weather, recent_level = self.catchment_data.all_current

        # Process
        Xs_current, y_current = BaseDataset.pre_process(current_weather, recent_level, allow_future_X=True)

        # Scale
        Xs_current = self.scaler.transform(Xs_current)
        y_current = self.target_scaler.transform(y_current)

        return (Xs_current, y_current)

    def update(self) -> None:
        """Update the underlying catchment data for inference with up to date data. This triggers new weather data queries and a rebuild of all current datasets."""
        self.Xs_current, self.y_current = self._get_data(update=True)
