from rlf.forecasting.base_forecaster import BaseForecaster

from darts.models.forecasting.forecasting_model import ForecastingModel


class TrainingForecaster(BaseForecaster):
    def __init__(self, model: ForecastingModel = None, **kwargs) -> None:
        super().__init__(**kwargs)

        self.model = model

    def fit(self) -> None:
        self.model.fit(self.dataset.y, past_covariates=self.dataset.Xs)
        self.model.save(self.model_save_path)
