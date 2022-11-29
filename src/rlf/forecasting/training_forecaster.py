import pickle

from darts.models.forecasting.forecasting_model import ForecastingModel

from rlf.forecasting.catchment_data import CatchmentData
from rlf.forecasting.base_forecaster import BaseForecaster, DEFAULT_WORK_DIR
from rlf.forecasting.training_dataset import TrainingDataset


class TrainingForecaster(BaseForecaster):
    """Forecaster abstraction for training models. Top level class interacted with by the user."""

    def __init__(
        self,
        model: ForecastingModel,
        dataset: TrainingDataset,
        catchment_data: CatchmentData,
        root_dir: str = DEFAULT_WORK_DIR,
        filename: str = "frcstr",
        scaler_filename: str = "scaler",
    ) -> None:
        """Create a training forecaster. Note that many important parameters must be passed as keyword args. See BaseForecaster docs for complete list.

        Args:
            model (ForecastingModel): A darts ForecastingModel to train.
            dataset (TrainingDataset): Dataset to use for training.
            catchment_data (CatchmentData): CatchmentData instance to use for training.
            root_dir (str, optional): Root dir to store trained model. Defaults to DEFAULT_WORK_DIR.
            filename (str, optional): Filename to use for trained model. Defaults to frcstr.
            scaler_filename (str, optional): Filename to use for the scalers. Defaults to "scaler".
        """
        super().__init__(
            catchment_data=catchment_data,
            root_dir=root_dir,
            filename=filename,
            scaler_filename=scaler_filename)

        self.model = model
        self.dataset = dataset

    def fit(self) -> None:
        """Fit the underlying Darts ForecastingModel model."""
        self.model.fit(self.dataset.y, future_covariates=self.dataset.X)
        self.save_model()

    def save_model(self) -> None:
        """Save the model and scalers to their specific paths."""
        self.model.save(self.model_save_path)

        scaler_map = {"scaler": self.dataset.scaler, "target_scaler": self.dataset.target_scaler}
        with open(self.scaler_save_path, "wb") as f:
            pickle.dump(scaler_map, f)
