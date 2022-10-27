from darts.models.forecasting.regression_ensemble_model import \
    RegressionEnsembleModel

from rlf.forecasting.base_forecaster import BaseForecaster

class InferenceForecaster(BaseForecaster):
    def __init__(self, model_type=RegressionEnsembleModel, **kwargs) -> None:
        super().__init__(**kwargs)

        self.model_type = model_type

    @property
    def model(self):
        return self._load_ensemble()

    def _load_ensemble(self):
        return self.model_type.load(self.ensemble_save_path)

    def predict(self, num_timesteps=24, update=False):
        if update:
            self.dataset.update()

        self.model.predict(num_timesteps, series=self.dataset.y, past_covariates=self.dataset.Xs)
