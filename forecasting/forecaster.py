from forecasting.dataset import Dataset
from forecasting.general_utilities.logging_utils import build_logger
from darts_custom.regression_ensemble_model_custom import RegressionEnsembleModelCustom


class Forecaster:
    """
    Highest level user facing class. Each instance is specific to a single gauge site/catchment.
    Managers training data, inference data, model fitting and inference of trained model.
    Allows the user to query for specific forecasts or forecast ranges.
    """
    def __init__(self, catchment_data, catchment_models, ensemble_train_size=0.2, likelihood_model=None,  log_level='INFO') -> None:
        """
        Fetches data from provided forecast site and generates processed training and inference sets.
        Builds the specified model.
        """
        self.logger = build_logger(log_level=log_level)
        self.name = catchment_data.name
        self.dataset = Dataset(catchment_data)

        self.catchment_models = catchment_models

        regression_train_n_points = int(ensemble_train_size * self.dataset.num_training_samples)
        self.ensemble_model = RegressionEnsembleModelCustom(catchment_models, likelihood_model, regression_train_n_points=regression_train_n_points)


    def fit(self):
        self.ensemble_model.fit(series=self.dataset.y_train, past_covariates=self.dataset.X_trains)


    def historical_forecasts(self, use_testing_holdout=False, **kwargs):
        """
        Create historical forecasts for the validation series using all models. Return a list of Timeseries
        """
        self.logger.info("Generating historical forecasts")
        if use_testing_holdout:
            y = self.dataset.y_test
            Xs = self.dataset.X_validations
        else:
            y = self.dataset.y_validation
            Xs = self.dataset.X_validations

        y_pred = self.ensemble_model.historical_forecasts(series=y, past_covariates=Xs, start=0.5, retrain=False, overlap_end=False, last_points_only=True, verbose=True , **kwargs)
        
        df = y_pred.pd_dataframe()
        return df


    def forecast_for_hours(self, n=24, num_samples=1):
        """
        """
        y = self.dataset.y_current
        Xs = self.dataset.Xs_current
        y_pred = self.ensemble_model.predict(n=n, series=y, past_covariates=Xs, num_samples=num_samples)
        df = y_pred.pd_dataframe
        return df


    def update_input_data(self) -> None:
        """
        Force an update to ensure inference data is up to date. Run at least hourly when forecasting.
        """
        self.prediction_set.update()