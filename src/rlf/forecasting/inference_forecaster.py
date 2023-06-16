import json
import os
import pickle
from typing import Mapping

from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.models.forecasting.forecasting_model import ForecastingModel
import numpy as np

from rlf.forecasting.base_forecaster import BaseForecaster, DEFAULT_WORK_DIR
from rlf.forecasting.catchment_data import CatchmentData
from rlf.forecasting.inference_dataset import InferenceDataset
from rlf.models.ensemble import Ensemble

def average_without_extremes(values):
    if len(values) <= 2:
        raise ValueError("The list should have at least 3 values.")

    sorted_values = sorted(values)
    trimmed_values = sorted_values[2:-2]  # Drop the first and last elements

    return sum(trimmed_values) / len(trimmed_values)


import math

def geometric_average(values):
    
    if len(values) <= 2:
        raise ValueError("The list should have at least 3 values.")

    trimmed_values = sorted(values)[1:-1]  # Drop the first and last elements

    # Calculate the product of the trimmed values
    product = math.prod(trimmed_values)

    # Calculate the geometric average by taking the nth root of the product, where n is the number of trimmed values
    geometric_avg = product ** (1 / len(trimmed_values))

    return geometric_avg



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


    def predict_average(self, num_timesteps: int = 24, update: bool = False) -> TimeSeries:
        """Generate a prediction that is the average of the predictions from each model in the ensemble. Drop the two largest and two smallest values at each step before averaging.

        Args:
            num_timesteps (int, optional): Number of timesteps to forecast. Defaults to 24.
            update (bool, optional): Whether or not to update the dataset. Defaults to False.

        Returns:
            TimeSeries: The prediction
        """
        if update:
            self.dataset.update()

        if (self.use_future_covariates):
            past_covariates = None
            future_covariates = self.dataset.X
        else:
            past_covariates = self.dataset.X
            future_covariates = None

        # Make the underlying contributing model predictions
        contributing_preds = self.model.predict_contributing_models(num_timesteps, series=self.dataset.y, past_covariates=past_covariates, future_covariates=future_covariates)

        # Convert the predictions to a single dataframe
        contributing_preds = contributing_preds.pd_dataframe()
        averages = contributing_preds.apply(average_without_extremes, axis=1)
        ts = TimeSeries.from_series(averages)
        # contributing_preds['averages'] = contributing_preds.apply(average_without_extremes, axis=1)
        
        # # Convert the averaged predictions to a TimeSeries
        rescaled_predictions = self.dataset.target_scaler.inverse_transform(ts)
        
        return rescaled_predictions

    def predict_geo_average(self, num_timesteps: int = 24, update: bool = False) -> TimeSeries:
        """Generate a prediction that is the average of the predictions from each model in the ensemble. Drop the two largest and two smallest values at each step before averaging.

        Args:
            num_timesteps (int, optional): Number of timesteps to forecast. Defaults to 24.
            update (bool, optional): Whether or not to update the dataset. Defaults to False.

        Returns:
            TimeSeries: The prediction
        """
        if update:
            self.dataset.update()

        if (self.use_future_covariates):
            past_covariates = None
            future_covariates = self.dataset.X
        else:
            past_covariates = self.dataset.X
            future_covariates = None

        # Make the underlying contributing model predictions
        contributing_preds = self.model.predict_contributing_models(num_timesteps, series=self.dataset.y, past_covariates=past_covariates, future_covariates=future_covariates)

        # Convert the predictions to a single dataframe
        contributing_preds = contributing_preds.pd_dataframe()
        averages = contributing_preds.apply(geometric_average, axis=1)
        ts = TimeSeries.from_series(averages)
        # contributing_preds['averages'] = contributing_preds.apply(average_without_extremes, axis=1)

        # # Convert the averaged predictions to a TimeSeries
        rescaled_predictions = self.dataset.target_scaler.inverse_transform(ts)

        return rescaled_predictions

    def predict_contributing_models(self, num_timesteps: int = 24, update: bool = False) -> TimeSeries:
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

        scaled_predictions = self.model.predict_contributing_models(num_timesteps, series=self.dataset.y, past_covariates=past_covariates, future_covariates=future_covariates)
        rescaled_predictions = self.dataset.target_scaler.inverse_transform(scaled_predictions)

        return rescaled_predictions



