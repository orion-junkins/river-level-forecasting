import json
import os
import pickle
from typing import Mapping

from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.models.forecasting.forecasting_model import ForecastingModel

from rlf.forecasting.base_forecaster import BaseForecaster, DEFAULT_WORK_DIR
from rlf.forecasting.catchment_data import CatchmentData
from rlf.forecasting.inference_dataset import InferenceDataset
from rlf.models.ensemble import Ensemble


class InferenceForecaster(BaseForecaster):
    """Forecaster abstraction for inference/production. Top level class interacted with by the user."""

    def __init__(
        self,
        catchment_data: CatchmentData,
        root_dir: str = DEFAULT_WORK_DIR,
        filename: str = "frcstr",
        load_cpu: bool = False
    ) -> None:
        """Create an inference forecaster.

        Args:
            catchment_data (CatchmentData): CatchmentData instance to use.
            root_dir (str, optional): Root directory where the model should be located. Defaults to DEFAULT_WORK_DIR.
            filename (str, optional): Name of file where the pickled model is located. Defaults to "frcstr".
            load_cpu (bool): If True then when loading the models set them to run inference on CPU. Defaults to False.
        """
        super().__init__(catchment_data=catchment_data, root_dir=root_dir, filename=filename)

        self._model = self._load_ensemble(load_cpu)

        scalers = self._load_scalers()
        metadata = self._load_metadata()
        catchment_data.columns = metadata["api_columns"]
        self.use_future_covariates = metadata["use_future_covariates"] if "use_future_covariates" in metadata else True

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
        return self._model

    def _load_ensemble(self, load_cpu: bool) -> ForecastingModel:
        """Load the underlying ForecastingModel.

        Args:
            load_cpu (bool): Whether or not to load models to run inference on CPU.

        Returns:
            ForecastingModel: Loaded ForecastingModel.
        """
        model = Ensemble.load(os.path.join(self.work_dir), load_cpu)
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
            TimeSeries: The forecasted timeseries rescaled to be in real units.
        """
        if update:
            self.dataset.update()

        if (self.use_future_covariates):
            past_covariates = None
            future_covariates = self.dataset.X
        else:
            past_covariates = self.dataset.X
            future_covariates = None

        scaled_predictions = self.model.predict(num_timesteps, series=self.dataset.y, past_covariates=past_covariates, future_covariates=future_covariates)
        rescaled_predictions = self.dataset.target_scaler.inverse_transform(scaled_predictions)

        return rescaled_predictions
