from rlf.forecasting.base_forecaster import BaseForecaster

from darts.models.forecasting.forecasting_model import ForecastingModel


class TrainingForecaster(BaseForecaster):
    """Forecaster abstraction for training models. Top level class interacted with by the user."""

    def __init__(self, model: ForecastingModel = None, **kwargs) -> None:
        """Create a training forecaster. Note that many important parameters must be passed as keyword args. See BaseForecaster docs for complete list.

        Args:
            model (ForecastingModel, optional): A darts ForecastingModel to train. Defaults to None.
        """
        super().__init__(**kwargs)

        self.model = model

    def fit(self) -> None:
        """Fit the underlying Darts ForecastingModel model."""
        self.model.fit(self.dataset.y, past_covariates=self.dataset.Xs)
        self.model.save(self.model_save_path)
