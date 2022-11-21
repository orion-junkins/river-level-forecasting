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
        filename: str = "frcstr"
    ) -> None:
        """Create a training forecaster. Note that many important parameters must be passed as keyword args. See BaseForecaster docs for complete list.

        Args:
            model (ForecastingModel): A darts ForecastingModel to train.
            dataset (TrainingDataset): Dataset to use for training.
            catchment_data (CatchmentData): CatchmentData instance to use for training.
            root_dir (str, optional): Root dir to store trained model. Defaults to DEFAULT_WORK_DIR.
            filename (str, optional): Filename to use for trained model. Defaults to frcstr.
        """
        super().__init__(catchment_data=catchment_data, root_dir=root_dir, filename=filename)

        self.model = model
        self.dataset = dataset

    def fit(self) -> None:
        """Fit the underlying Darts ForecastingModel model."""
        self.model.fit(self.dataset.y, past_covariates=self.dataset.Xs)
        self.model.save(self.model_save_path)
