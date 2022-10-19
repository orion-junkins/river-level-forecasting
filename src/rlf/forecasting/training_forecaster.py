from rlf.forecasting.forecaster_abc import Forecaster_ABC
from rlf.forecasting.training_dataset import TrainingDataset


class TrainingForecaster(Forecaster_ABC):
    def __init__(self, model, **kwargs) -> None:
        super().__init__(**kwargs)

        self.model = model
        self.dataset = TrainingDataset(catchment_data=self.catchment_data)

    def fit(self):
        self.model.fit(self.dataset.y, past_covariates=self.dataset.Xs)
        self.model.save(self.model_save_path)
