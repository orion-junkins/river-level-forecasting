import sys
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
        self.ensemble_model = RegressionEnsembleModelCustom(catchment_models, likelihood_model, regression_train_n_points=regression_train_n_points)


    def fit(self):
        self.ensemble_model.fit(series=self.dataset.y_train, past_covariates=self.dataset.X_trains)


    def historical_forecasts(self, data_partition="validation", **kwargs):
        self.logger.info("Generating historical forecasts")
        y = self.get_y(data_partition)
        Xs = self.get_Xs(data_partition)
        target_scaler = self.dataset.target_scaler

        y_pred = self.ensemble_model.historical_forecasts(series=y, past_covariates=Xs, start=0.05, retrain=False, overlap_end=False, last_points_only=True, verbose=True , **kwargs)
        y_pred = target_scaler.inverse_transform(y_pred)
        df = y_pred.pd_dataframe()
        if "level" in df.columns:
            print("ensembled is level!")
            df.rename(columns={"level": "level_pred"}, inplace=True)
        elif "0" in df.columns:
            print("ensembled is 0!")
            df.rename(columns={"0": "level_pred"}, inplace=True)
        return df

    def raw_historical_forecasts(self, data_partition="validation", **kwargs):
        self.logger.info("Generating historical forecasts")
        y = self.get_y(data_partition)
        Xs = self.get_Xs(data_partition)

        models = self.ensemble_model.models
        target_scaler = self.dataset.target_scaler
        y_preds = []

        for index, (X, model) in enumerate(zip(Xs, models)):
            y_pred = model.historical_forecasts(series=y, past_covariates=X, start=0.5, retrain=False, overlap_end=False, last_points_only=True, verbose=True , **kwargs)
            y_pred = target_scaler.inverse_transform(y_pred)
            y_pred = y_pred.pd_dataframe()
            if "level" in y_pred.columns:
                print("raw is level!")
                y_pred.rename(columns={"level": "level_"+str(index)}, inplace=True)
            elif "0" in y_pred.columns:
                print("raw is 0!")
                y_pred.rename(columns={"0": "level_"+str(index)}, inplace=True)
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

        y_pred_ensembled_df.rename(columns={"level": "level_pred"}, inplace=True)
        return y_pred_ensembled_df


    def raw_forecast_for_hours(self, n=24, num_samples=1):
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

    def get_forecast(self, hours_to_forecast=24, update_dataset=True):
        if update_dataset:
            self.update_dataset()
            
        # Generate forecast
        y_ensembled_forecasted = self.forecast_for_hours(hours_to_forecast)
        y_raw_forecasted = self.raw_forecast_for_hours(hours_to_forecast)
        
        # Grab recent data
        y_recent = self.get_y_df(data_partition="current")

        # Join all data into single dataframe
        frames = [y_recent, y_raw_forecasted, y_ensembled_forecasted]
        df = pd.concat(frames, axis=1)

        return df
        
    def get_historical(self, data_partition="validation", **kwargs):
        y_raw_hst_fcasts = self.raw_historical_forecasts(data_partition=data_partition, **kwargs)
        y_ensembled_hst_fcasts = self.historical_forecasts(data_partition=data_partition, **kwargs)
        
        # Grab recent data
        y_true = self.get_y_df(data_partition=data_partition)

        # Join all data into single dataframe
        frames = [y_raw_hst_fcasts, y_ensembled_hst_fcasts, y_true]
        df = pd.concat(frames, axis=1)

        df.rename(columns={"index": "datetime"})
        return df
        
   
    def update_dataset(self) -> None:
        """
        Force an update to ensure inference data is up to date. Run at least hourly when forecasting.
        """
        self.dataset.update()

    def get_y(self, data_partition):
        if data_partition == "current":
            y = self.dataset.y_current
        elif data_partition == "train":
            y = self.dataset.y_train
        elif data_partition == "validation":
            y = self.dataset.y_validation
        elif data_partition == "test":
            y = self.dataset.y_test
        else:
            self.logger.error("Invalid argument")
            sys.exit()
        return y
    
    def get_y_df(self, data_partition):
        y = self.get_y(data_partition)
        target_scaler = self.dataset.target_scaler
        y = target_scaler.inverse_transform(y)
        df_y = y.pd_dataframe()
        df_y.rename(columns={"level": "level_true"}, inplace=True)
        return df_y
    
    def get_Xs(self, data_partition):
        if data_partition == "current":
            Xs = self.dataset.Xs_current
        elif data_partition == "train":
            Xs = self.dataset.Xs_train
        elif data_partition == "validation":
            Xs = self.dataset.Xs_validation
        elif data_partition == "test":
            Xs = self.dataset.Xs_test
        else:
            self.logger.error("Invalid argument")
            sys.exit()
        
        return Xs