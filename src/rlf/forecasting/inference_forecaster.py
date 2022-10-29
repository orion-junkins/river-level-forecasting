from darts import timeseries
from darts.models.forecasting.regression_ensemble_model import \
    RegressionEnsembleModel
from darts.models.forecasting.forecasting_model import ForecastingModel


from rlf.forecasting.base_forecaster import BaseForecaster


class InferenceForecaster(BaseForecaster):
    """Forecaster abstraction for inference/production. Top level class interacted with by the user."""
    def __init__(self, model_type: ForecastingModel = RegressionEnsembleModel, **kwargs) -> None:
        """Create a training forecaster. Note that many important parameters must be passed as keyword args. See BaseForecaster docs for complete list.


        Args:
            model_type (ForecastingModel, optional): Darts Forecasting model type to load. Defaults to RegressionEnsembleModel.
        """
        super().__init__(**kwargs)

        self.model_type = model_type

    @property
    def model(self) -> ForecastingModel:
        """A loaded ForecastingModel. Expected to be fully trained.

        Returns:
            ForecastingModel: Loaded ForecastingModel.
        """
        return self._load_ensemble()

    def _load_ensemble(self) -> ForecastingModel:
        """Load the underlying ForecastingModel.

        Returns:
            ForecastingModel: Loaded ForecastingModel.
        """
        return self.model_type.load(self.ensemble_save_path)

    def predict(self, num_timesteps: int = 24, update: bool = False) -> timeseries:
        """Generate a prediction.

        Args:
            num_timesteps (int, optional): Number of timesteps into the future to predict. Defaults to 24.
            update (bool, optional): Whether or not to update underlying Dataset before inference. Defaults to False.

        Returns:
            timeseries: The forecasted timeseries.
        """
        if update:
            self.dataset.update()

        return self.model.predict(num_timesteps, series=self.dataset.y, past_covariates=self.dataset.Xs)
