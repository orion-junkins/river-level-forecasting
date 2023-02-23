import json
import os
import pickle
from typing import Mapping

from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.models.forecasting.forecasting_model import ForecastingModel
from darts.models.forecasting.regression_ensemble_model import (
    RegressionEnsembleModel
)

from rlf.forecasting.base_forecaster import BaseForecaster, DEFAULT_WORK_DIR
from rlf.forecasting.catchment_data import CatchmentData
from rlf.forecasting.inference_dataset import InferenceDataset
from rlf.models.utils import load_ensemble_model


class InferenceForecaster(BaseForecaster):
    """Forecaster abstraction for inference/production. Top level class interacted with by the user."""
    def __init__(
        self,
        catchment_data: CatchmentData,
        root_dir: str = DEFAULT_WORK_DIR,
        filename: str = "frcstr",
    ) -> None:
        """Create a training forecaster. Note that many important parameters must be passed as keyword args. See BaseForecaster docs for complete list.

        Args:
            catchment_data (CatchmentData): CatchmentData instance to use.
            root_dir (str, optional): Root directory where the model should be located. Defaults to DEFAULT_WORK_DIR.
            filename (str, optional): Name of file where the pickled model is located. Defaults to "frcstr".
        """
        super().__init__(catchment_data=catchment_data, root_dir=root_dir, filename=filename)

        self._model = self._load_ensemble()

        scalers = self._load_scalers()
        metadata = self._load_metadata()
        catchment_data.columns = metadata["api_columns"]

        self.dataset = InferenceDataset(
            scalers["scaler"],
            scalers["target_scaler"],
            catchment_data,
            rolling_sum_columns=metadata["sum_columns"],
            rolling_mean_columns=metadata["mean_columns"],
            rolling_window_sizes=metadata["windows"]
        )

    @property
    def model(self) -> ForecastingModel:
        """A loaded ForecastingModel. Expected to be fully trained.

        Returns:
            ForecastingModel: Loaded ForecastingModel.
        """
        return self._load_ensemble()

    def _load_ensemble(self) -> ForecastingModel:
        """Load the underlying ForecastingModel.
        Returns:
            ForecastingModel: Loaded ForecastingModel.
        """
        model = load_ensemble_model(self.work_dir)
        return model

    def _load_scalers(self) -> Mapping[str, Scaler]:
        with open(self.scaler_save_path, "rb") as f:
            scalers = pickle.load(f)
        return scalers

    def _load_metadata(self) -> dict:
        with open(os.path.join(self.work_dir, "metadata")) as f:
            metadata = json.load(f)
        return metadata

    def predict(self, num_timesteps: int = 24, update: bool = False) -> TimeSeries:
        """Generate a prediction.

        Args:
            num_timesteps (int, optional): Number of timesteps into the future to predict. Defaults to 24.
            update (bool, optional): Whether or not to update underlying Dataset before inference. Defaults to False.

        Returns:
            TimeSeries: The forecasted timeseries.
        """
        if update:
            self.dataset.update()

        return self.model.predict(num_timesteps, series=self.dataset.y, past_covariates=self.dataset.Xs)
