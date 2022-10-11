from forecasting.forecaster_abc import Forecaster_ABC
from forecasting.inference_dataset import InferenceDataset
from darts.models.forecasting.regression_ensemble_model import RegressionEnsembleModel

class InferenceForecaster(Forecaster_ABC):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        
        self.ensemble = self.load_ensemble()
        self.dataset= InferenceDataset(catchment_data=self.catchment_data)
    
    def load_ensemble(self):
        RegressionEnsembleModel.load(self.ensemble_save_path)


    def predict(self, num_timesteps=24):
        self.ensemble.predict(num_timesteps, series=self.dataset.y, past_covariates=self.dataset.Xs)

