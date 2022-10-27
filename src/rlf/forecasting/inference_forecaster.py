from darts import timeseries
from darts.models.forecasting.regression_ensemble_model import \
    RegressionEnsembleModel
from darts.models.forecasting.forecasting_model import ForecastingModel

from rlf.forecasting.base_forecaster import BaseForecaster


class InferenceForecaster(BaseForecaster):
    """"""
    def __init__(self, model_type: ForecastingModel = RegressionEnsembleModel, **kwargs) -> None:
        super().__init__(**kwargs)

        self.model_type = model_type

    @property
    def model(self) -> ForecastingModel:
        return self._load_ensemble()

    def _load_ensemble(self) -> ForecastingModel:
        return self.model_type.load(self.ensemble_save_path)

    def predict(self, num_timesteps: int = 24, update: bool = False) -> timeseries:
        if update:
            self.dataset.update()

        return self.model.predict(num_timesteps, series=self.dataset.y, past_covariates=self.dataset.Xs)
