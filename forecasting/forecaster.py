import pandas as pd
from forecasting.dataset import Dataset
from forecasting.general_utilities.logging_utils import build_logger
from darts_custom.regression_ensemble_model_custom import RegressionEnsembleModelCustom


class Forecaster:
    """
    Highest level user facing class. Each instance is specific to a single gauge site/catchment.
    Managers training data, inference data, model fitting and inference of trained model.
    Allows the user to query for specific forecasts or forecast ranges.
    """
    def __init__(self, catchment_data, catchment_models, regression_model=None, ensemble_train_size=0.3, likelihood_model=None,  log_level='INFO') -> None:
        """
        Fetches data from provided forecast site and generates processed training and inference sets.
        Builds the specified model.
        """
        self.logger = build_logger(log_level=log_level)
        self.name = catchment_data.name
        self.dataset = Dataset(catchment_data)

        self.catchment_models = catchment_models

        regression_train_n_points = int(ensemble_train_size * self.dataset.num_training_samples)
        self.ensemble_model = RegressionEnsembleModelCustom(catchment_models, likelihood_model, regression_model=regression_model, regression_train_n_points=regression_train_n_points)


    def fit(self):
        self.ensemble_model.fit(series=self.dataset.y_train, past_covariates=self.dataset.X_trains)


    def historical_forecasts(self, use_testing_holdout=False, **kwargs):
        self.logger.info("Generating historical forecasts")
        if use_testing_holdout:
            y = self.dataset.y_test
            Xs = self.dataset.X_tests
        else:
            y = self.dataset.y_validation
            Xs = self.dataset.X_validations

        y_pred = self.ensemble_model.historical_forecasts(series=y, past_covariates=Xs, start=0.5, retrain=False, overlap_end=False, last_points_only=True, verbose=True , **kwargs)
        
        df = y_pred.pd_dataframe()
        return df

    def raw_historical_forecasts(self, use_testing_holdout=False, **kwargs):
        self.logger.info("Generating historical forecasts")
        if use_testing_holdout:
            y = self.dataset.y_test
            Xs = self.dataset.X_tests
        else:
            y = self.dataset.y_validation
            Xs = self.dataset.X_validations

        models = self.ensemble_model.models
        target_scaler = self.dataset.target_scaler
        y_preds = []

        for index, (X, model) in enumerate(zip(Xs, models)):
            y_pred = model.historical_forecasts(series=y, past_covariates=X, start=0.95, retrain=False, overlap_end=False, last_points_only=True, verbose=True , **kwargs)
            y_pred = target_scaler.inverse_transform(y_pred)
            y_pred = y_pred.pd_dataframe()
            y_pred.rename(columns={"level": "level_"+str(index)}, inplace=True)
            y_preds.append(y_pred)

        df_y_preds = pd.concat(y_preds, axis=1)
        return df_y_preds


    def forecast_for_hours(self, n=24, num_samples=1):
        """
        """
        y = self.dataset.y_current
        Xs = self.dataset.Xs_current
        y_pred_ensembled = self.ensemble_model.predict(n=n, series=y, past_covariates=Xs, num_samples=num_samples)

        target_scaler = self.dataset.target_scaler

        y_pred_ensembled = target_scaler.inverse_transform(y_pred_ensembled)
        
        y_pred_ensembled_df = y_pred_ensembled.pd_dataframe()
        return y_pred_ensembled_df


    def raw_forecasts_for_hours(self, n=24, num_samples=1):
        """
        """
        y = self.dataset.y_current
        Xs = self.dataset.Xs_current
        models = self.ensemble_model.models
        target_scaler = self.dataset.target_scaler
        y_preds = []

        for index, (X, model) in enumerate(zip(Xs, models)):
            y_pred = model.predict(n=n, series=y, past_covariates=X, num_samples=num_samples)
            y_pred = target_scaler.inverse_transform(y_pred)
            y_pred = y_pred.pd_dataframe()
            y_pred.rename(columns={"level": "level_"+str(index)}, inplace=True)
            y_preds.append(y_pred)

        df_y_preds = pd.concat(y_preds, axis=1)
        return df_y_preds

    def update_input_data(self) -> None:
        """
        Force an update to ensure inference data is up to date. Run at least hourly when forecasting.
        """
        self.dataset.update()