import json
import os
import pickle

from darts.metrics.metrics import mae
from darts.timeseries import TimeSeries

import numpy as np
from rlf.forecasting.base_forecaster import BaseForecaster, DEFAULT_WORK_DIR
from rlf.forecasting.data_fetching_utilities.weather_provider.base_weather_provider import BaseWeatherProvider
from rlf.forecasting.inference_forecaster import InferenceForecaster
from rlf.forecasting.training_dataset import TrainingDataset
from rlf.models.ensemble import Ensemble


class TrainingForecaster(BaseForecaster):
    """Forecaster abstraction for training models. Top level class interacted with by the user."""

    def __init__(
        self,
        model: Ensemble,
        dataset: TrainingDataset,
        root_dir: str = DEFAULT_WORK_DIR,
        scaler_filename: str = "scaler",
        use_future_covariates: bool = True
    ) -> None:
        """Create a training forecaster. Note that many important parameters must be passed as keyword args. See BaseForecaster docs for complete list.

        Args:
            model (ForecastingModel): A darts ForecastingModel to train.
            dataset (TrainingDataset): TrainingDataset instance to use for training.
            root_dir (str, optional): Root dir to store trained model. Defaults to DEFAULT_WORK_DIR.
            scaler_filename (str, optional): Filename to use for the scalers. Defaults to "scaler".
            use_future_covariates (bool, optional): Whether to use data as future covariates or past covariates. Defaults to True.
        """
        super().__init__(
            catchment_data=dataset.catchment_data,
            root_dir=root_dir,
            filename="frcstr",
            scaler_filename=scaler_filename)

        self.model = model
        self.dataset = dataset
        self.use_future_covariates = use_future_covariates

        os.makedirs(self.work_dir, exist_ok=True)
        if (os.path.isfile(self.model_save_path)):
            raise ValueError(f"{self.model_save_path} already exists. Specify a unique save path.")

        if (os.path.isfile(self.scaler_save_path)):
            raise ValueError(f"{self.scaler_save_path} already exists. Specify a unique save path.")

    def fit(self) -> None:
        """Fit the underlying Darts ForecastingModel model."""
        self.model.fit_dataset(self.dataset, use_future_covariates=self.use_future_covariates, retrain_contributing_models=True)
        self.save_model()

    def save_model(self) -> None:
        """Save the model and scalers to their specific paths."""
        # save_ensemble_model(self.work_dir, self.model)
        self.model.save(self.work_dir)

        scaler_map = {"scaler": self.dataset.scaler, "target_scaler": self.dataset.target_scaler}
        with open(self.scaler_save_path, "wb") as f:
            pickle.dump(scaler_map, f)

        # dump the metadata
        metadata = {
            "api_columns": self.dataset.base_columns,
            "engineered_columns": ["day_of_year"],
            "mean_columns": self.dataset.rolling_mean_columns,
            "sum_columns": self.dataset.rolling_sum_columns,
            "windows": self.dataset.rolling_window_sizes,
            "use_future_covariates": self.use_future_covariates,
        }

        with open(os.path.join(self.work_dir, "metadata"), "w") as f:
            json.dump(metadata, f)

    def backtest(self,
                 run_on_validation: bool = False,
                 retrain: bool = False,
                 start: float = 0.05,  # Data must extend slightly before start. Increase value or use a larger dataset if "" error occurs.
                 forecast_horizon: int = 24,
                 stride: int = 24,
                 last_points_only: bool = False) -> float:
        """Backtest the model on the training data. This is useful for debugging and hyperparameter tuning. See darts docs for more details:
        https://unit8co.github.io/darts/generated_api/darts.models.forecasting.regression_ensemble_model.html#darts.models.forecasting.regression_ensemble_model.RegressionEnsembleModel.backtest

        Args:
            run_on_validation (bool, optional): Whether to run on the validation dataset. Runs on Test set if False. Defaults to False.
            future_covariates (bool, optional): Whether to pass X as future covariates. X will be passed as past_covariates if False. Defaults to True.
            retrain (bool, optional): Whether to retrain the model on the entire training dataset. Defaults to False.
            start (float, optional): The proportion of the training dataset to use for backtesting. Defaults to 0.95.
            forecast_horizon (int, optional): The forecast horizon to use. Defaults to 24.
            stride (int, optional): The stride to use. Defaults to 24.
            last_points_only (bool, optional): Whether to only use the last point in the training dataset. Defaults to False.

        Returns:
            float: The backtest score. See darts docs for more details.
        """
        if (run_on_validation):
            x = self.dataset.X_validation
            y = self.dataset.y_validation
        else:
            x = self.dataset.X_test
            y = self.dataset.y_test

        if (self.use_future_covariates):
            past_covariates = None
            future_covariates = x
        else:
            past_covariates = x
            future_covariates = None

        error = self.model.backtest(y,
                                    past_covariates=past_covariates,
                                    future_covariates=future_covariates,
                                    retrain=retrain,
                                    start=start,
                                    forecast_horizon=forecast_horizon,
                                    stride=stride,
                                    last_points_only=last_points_only,
                                    metric=mae)
        scaled_error = np.array([[error]], np.float32)
        scaled_error_ts = TimeSeries.from_values(scaled_error)
        unscaled_error = float(self.dataset.target_scaler.inverse_transform(scaled_error_ts).values()[0][0])
        return unscaled_error

    def backtest_contributing_models(self,
                                     run_on_validation: bool = False,
                                     retrain: bool = False,
                                     start: float = 0.05,  # Data must extend slightly before start. Increase value or use a larger dataset if "" error occurs.
                                     forecast_horizon: int = 24,
                                     stride: int = 24,
                                     last_points_only: bool = False) -> list:
        """Backtest the underlying contributing models on the training data. Return the average of all error individual model scores. This is useful for debugging and hyperparameter tuning. See darts docs for more details:
        https://unit8co.github.io/darts/generated_api/darts.models.forecasting.regression_ensemble_model.html#darts.models.forecasting.regression_ensemble_model.RegressionEnsembleModel.backtest

        Args:
            run_on_validation (bool, optional): Whether to run on the validation dataset. Runs on Test set if False. Defaults to False.
            retrain (bool, optional): Whether to retrain the model on the entire training dataset. Defaults to False.
            start (float, optional): The proportion of the training dataset to use for backtesting. Defaults to 0.05.
            forecast_horizon (int, optional): The forecast horizon to use. Defaults to 24.
            stride (int, optional): The stride to use. Defaults to 24.
            last_points_only (bool, optional): Whether to only use the last point in the training dataset. Defaults to False.

        Returns:
            float: The backtest score. See darts docs for more details.
        """
        if (run_on_validation):
            x = self.dataset.X_validation
            y = self.dataset.y_validation
        else:
            x = self.dataset.X_test
            y = self.dataset.y_test

        if (self.use_future_covariates):
            past_covariates = None
            future_covariates = x
        else:
            past_covariates = x
            future_covariates = None

        model_errors = []
        for model in self.model.contributing_models:
            error = model.backtest(y,
                                   past_covariates=past_covariates,
                                   future_covariates=future_covariates,
                                   retrain=retrain,
                                   start=start,
                                   forecast_horizon=forecast_horizon,
                                   stride=stride,
                                   last_points_only=last_points_only,
                                   metric=mae)
            scaled_error = np.array([[error]], np.float32)
            scaled_error_ts = TimeSeries.from_values(scaled_error)
            unscaled_error = float(self.dataset.target_scaler.inverse_transform(scaled_error_ts).values()[0][0])
            model_errors.append(unscaled_error)

        return model_errors


def load_training_forecaster(loaded_inference_forecaster: InferenceForecaster, weather_provider: BaseWeatherProvider) -> TrainingForecaster:
    """Given a loaded inference forecaster, extract the trained models and build a training forecaster.

    Args:
        loaded_inference_forecaster (InferenceForecaster): Loaded Inference Forecaster with trained models
        weather_provider (BaseWeatherProvider): Weather Provider to use for training forecaster

    Returns:
        TrainingForecaster: Loaded Training Forecaster with models from Inference Forecaster.
    """
    # Isolate needed data from loaded inference forecaster
    catchment_name = loaded_inference_forecaster.catchment_data.name
    level_provider = loaded_inference_forecaster.catchment_data.level_provider
    columns = loaded_inference_forecaster.catchment_data.columns
    root_dir = os.path.join("loaded_trained_model", loaded_inference_forecaster.root_dir)
    model = loaded_inference_forecaster.model

    # Build a new CatchmentData object with the same data and the provided weather provider
    catchment_data = CatchmentData(
        catchment_name,
        weather_provider,
        level_provider,
        columns=columns
    ) 

    # Build a new TrainingForecaster object with the same model and the new CatchmentData object
    dataset = TrainingDataset(catchment_data)
    training_forecaster = TrainingForecaster(model, dataset, root_dir=root_dir)

    return training_forecaster