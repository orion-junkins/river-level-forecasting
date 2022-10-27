from rlf.forecasting.base_forecaster import BaseForecaster


class TrainingForecaster(BaseForecaster):
    def __init__(self, model=None, **kwargs) -> None:
        super().__init__(**kwargs)

        self.model = model

    def fit(self):
        self.model.fit(self.dataset.y, past_covariates=self.dataset.Xs)
        self.model.save(self.model_save_path)
