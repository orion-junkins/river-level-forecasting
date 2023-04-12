import json
import os
import pickle

from darts.models.forecasting.forecasting_model import ForecastingModel

from rlf.forecasting.base_forecaster import BaseForecaster, DEFAULT_WORK_DIR
from rlf.forecasting.training_dataset import TrainingDataset
from rlf.models.utils import save_ensemble_model
from rlf.models.ensemble import Ensemble


class TrainingForecaster(BaseForecaster):
    """Forecaster abstraction for training models. Top level class interacted with by the user."""

    def __init__(
        self,
        model: Ensemble,
        dataset: TrainingDataset,
        root_dir: str = DEFAULT_WORK_DIR,
        scaler_filename: str = "scaler",
    ) -> None:
        """Create a training forecaster. Note that many important parameters must be passed as keyword args. See BaseForecaster docs for complete list.

        Args:
            model (ForecastingModel): A darts ForecastingModel to train.
            dataset (TrainingDataset): TrainingDataset instance to use for training.
            root_dir (str, optional): Root dir to store trained model. Defaults to DEFAULT_WORK_DIR.
            scaler_filename (str, optional): Filename to use for the scalers. Defaults to "scaler".
        """
        super().__init__(
            catchment_data=dataset.catchment_data,
            root_dir=root_dir,
            filename="frcstr",
            scaler_filename=scaler_filename)

        self.model = model
        self.dataset = dataset

        os.makedirs(self.work_dir, exist_ok=True)
        if (os.path.isfile(self.model_save_path)):
            raise ValueError(f"{self.model_save_path} already exists. Specify a unique save path.")

        if (os.path.isfile(self.scaler_save_path)):
            raise ValueError(f"{self.scaler_save_path} already exists. Specify a unique save path.")

    def fit(self) -> None:
        """Fit the underlying Darts ForecastingModel model."""
        self.model.fit(self.dataset.y_train, future_covariates=self.dataset.X_train, retrain_contributing_models=True)
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
            "windows": self.dataset.rolling_window_sizes
        }

        with open(os.path.join(self.work_dir, "metadata"), "w") as f:
            json.dump(metadata, f)

    def backtest(self,
                 run_on_validation: bool = False,
                 future_covariates: bool = True,
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

        if (future_covariates):
            past_covariates_data = None
            future_covariates_data = x
        else:
            past_covariates_data = x
            future_covariates_data = None

        return self.model.backtest(y,
                                   past_covariates=past_covariates_data,
                                   future_covariates=future_covariates_data,
                                   retrain=retrain,
                                   start=start,
                                   forecast_horizon=forecast_horizon,
                                   stride=stride,
                                   last_points_only=last_points_only)

    def backtest_contributing_models(self,
                                     run_on_validation: bool = False,
                                     future_covariates: bool = True,
                                     retrain: bool = False,
                                     start: float = 0.05,  # Data must extend slightly before start. Increase value or use a larger dataset if "" error occurs.
                                     forecast_horizon: int = 24,
                                     stride: int = 24,
                                     last_points_only: bool = False) -> list:
        """Backtest the underlying contributing models on the training data. Return the average of all error individual model scores. This is useful for debugging and hyperparameter tuning. See darts docs for more details:
        https://unit8co.github.io/darts/generated_api/darts.models.forecasting.regression_ensemble_model.html#darts.models.forecasting.regression_ensemble_model.RegressionEnsembleModel.backtest

        Args:
            run_on_validation (bool, optional): Whether to run on the validation dataset. Runs on Test set if False. Defaults to False.
            future_covariates (bool, optional): Whether to pass X as future covariates. X will be passed as past_covariates if False. Defaults to True.
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

        if (future_covariates):
            past_covariates_data = None
            future_covariates_data = x
        else:
            past_covariates_data = x
            future_covariates_data = None

        model_errors = []
        for model in self.model.models:
            model_errors.append(model.backtest(y,
                                               past_covariates=past_covariates_data,
                                               future_covariates=future_covariates_data,
                                               retrain=retrain,
                                               start=start,
                                               forecast_horizon=forecast_horizon,
                                               stride=stride,
                                               last_points_only=last_points_only))

        return model_errors
