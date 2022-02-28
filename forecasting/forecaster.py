import os
import numpy as np
import pandas as pd
from datetime import timedelta

from darts.models import BlockRNNModel
from darts.utils.likelihood_models import GaussianLikelihood
from forecasting.dataset import Dataset
from forecasting.general_utilities.logging_utils import build_logger


class Forecaster:
    """
    Highest level user facing class. Each instance is specific to a single gauge site/catchment.
    Managers training data, inference data, model fitting and inference of trained model.
    Allows the user to query for specific forecasts or forecast ranges.
    """
    def __init__(self, catchment_data, model_type=BlockRNNModel, model_params={}, overwrite_existing_models=False, log_level='INFO', parent_dir="trained_models", model_save_dir="model", likelihood=GaussianLikelihood(), verbose=True) -> None:
        """
        Fetches data from provided forecast site and generates processed training and inference sets.
        Builds the specified model.
        """
        self.logger = build_logger(log_level=log_level)
        self.likelihood = likelihood
        self.name = catchment_data.name
        self.dataset = Dataset(catchment_data)

        self.model_builder = model_type
        self.model_params = model_params
        self.model_save_dir = os.path.join(parent_dir, self.name, model_save_dir)
        self.models=[]

        if overwrite_existing_models or not self.checkpoint_dir_exists():
            os.makedirs(self.model_save_dir, exist_ok=True)
            self.models = self._build_new_models()
        else:
            self.models = self._load_existing_models()
            

    def checkpoint_dir_exists(self):
        checkpoint_dir = os.path.join(self.model_save_dir, "checkpoints")
        return os.path.exists(checkpoint_dir)


    def _load_existing_models(self):
        self.logger.info("Loading existing models.")
        models = []
        for index in range(self.dataset.num_X_sets):
            self.logger.info("Loading model for dataset %s" % index)
            model = self.model_builder.load_from_checkpoint(str(index), work_dir=self.model_save_dir)
            models.append(model)
        self.logger.info("All models loaded!")
        return models


    def _build_new_models(self):
        self.logger.info("Building new models. Overwritting any existing models.")
        models = []
        for index in range(self.dataset.num_X_sets):
            self.logger.info("Building model for dataset %s" % index)
            model = self.model_builder(**self.model_params, 
                                        work_dir=self.model_save_dir, model_name=str(index), 
                                        force_reset=True, save_checkpoints=True,
                                        likelihood=self.likelihood)
            models.append(model)
        self.logger.info("All models built!")
        return models

        
    def fit(self, **kwargs):
        """
        Wrapper for tf.keras.model.fit() to train internal model instance. Exposes select tuning parameters.

        Args:
            epochs (int, optional):  Number of epochs to train the model. Defaults to 20.
            batch_size (int, optional): Number of samples per gradient update. Defaults to 10.
            shuffle (bool, optional): Whether to shuffle the training data before each epoch. Defaults to True.
        """
        self.logger.info("Fitting all models")

        models = self.models
        X_trains = self.dataset.X_trains
        X_validations = self.dataset.X_validations
        y_train = self.dataset.y_train
        y_val= self.dataset.y_validation
        
        for index, (model, X_train, X_val) in enumerate(zip(models, X_trains, X_validations)):
            self.logger.info("Fitting model %s" % index)
            model.fit(series=y_train, past_covariates=X_train, 
                        val_series=y_val, val_past_covariates=X_val, 
                        verbose=True, **kwargs)


    def forecast_for_range(self, start, end):
        cur_timestep = start
        while cur_timestep < end:
            level = self.forecast_for(cur_timestep)
            self.prediction_set.record_level_for(cur_timestep, level)
            cur_timestep = cur_timestep + timedelta(hours=1)


    def forecast_for(self, timestamp):
        """
        Run inference for the given timestamp.

        Args:
            timestamp (datetime): The date & time for which a forecast is desired. Must be a member of PredictionSet indices.

        Returns:
            y_pred (float): The predicted river level at the given timestamp
        """
        x_in = self.prediction_set.x_in_for_window(timestamp)
        x_in = np.array([x_in])
        y_pred = np.array(self.model.predict(x_in))

        # Inverse transform result
        target_scaler = self.dataset.target_scaler
        y_pred = target_scaler.inverse_transform(y_pred)

        # Convert to float from np.array
        y_pred = y_pred[0][0]  
        return y_pred


    def historical_forecasts(self, **kwargs):
        """
        Create historical forecasts for the validation series using all models. Return a list of Timeseries
        """
        self.logger.info("Generating historical forecasts")
        y_preds_min = []
        y_preds_mid = []
        y_preds_max = []

        target_scaler = self.dataset.target_scaler
        y_val = self.dataset.y_validation
        for index, (model, X_val) in enumerate(zip(self.models, self.dataset.X_validations)):
            self.logger.info("Generating historical forecast for model %s" % index)
            y_pred = model.historical_forecasts(series=y_val, past_covariates=X_val, start=0.5, retrain=False, overlap_end=False, last_points_only=True, verbose=True , **kwargs)
            y_pred = target_scaler.inverse_transform(y_pred)
            if y_pred.is_stochastic:
                y_pred_min = y_pred.quantile_df(0.05).applymap(lambda x: x.item())
                y_preds_min.append(y_pred_min)
                y_pred_mid = y_pred.quantile_df(0.5).applymap(lambda x: x.item())
                y_preds_mid.append(y_pred_mid)
                y_pred_max = y_pred.quantile_df(0.95).applymap(lambda x: x.item())
                y_preds_max.append(y_pred_max)
            else:
                y_preds_mid.append(y_pred)
        return (y_preds_min, y_preds_mid, y_preds_max)


    def _join_preds(self, y_preds):
        """
        Consolidate a list of predicted timeseries into a single dataframe 

        Args:
            y_preds (_type_): _description_
        """
        df = pd.DataFrame()
        for index, y_pred in enumerate(y_preds):
            if y_pred.is_stochastic:
                df_cur = y_pred.quantiles_df(quantiles=(0.05, 0.5, 0.95))
            else:
                df_cur = y_pred.pd_dataframe()
            df_cur = df_cur.rename(columns={"0": str(index)})
            frames = [df, df_cur]
            df = pd.concat(frames, axis=1)
        return df


    def _inverse_scale_all(self, y_preds):
        """
        scaled list of series -> inverse scaled list of series

        Args:
            y_preds (_type_): _description_

        Returns:
            _type_: _description_
        """
        inverse_scaled_y_preds = []
        target_scaler = self.dataset.target_scaler
        for series in y_preds:
            series = target_scaler.inverse_transform(series)
            inverse_scaled_y_preds.append(series)
        return inverse_scaled_y_preds

    def update_input_data(self) -> None:
        """
        Force an update to ensure inference data is up to date. Run at least hourly when forecasting.
        """
        self.prediction_set.update()