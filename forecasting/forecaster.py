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
    def __init__(self, catchment_data, model_type=BlockRNNModel, model_params={}, overwrite_existing_models=False, log_level='INFO', parent_dir="trained_models", model_save_dir="model", likelihood=None, verbose=True) -> None:
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

        if overwrite_existing_models:
            self.merged_model = self.model_builder(**self.model_params, 
                                        work_dir=self.model_save_dir, model_name=str("merged"), 
                                        force_reset=True, save_checkpoints=True,
                                        likelihood=self.likelihood,
                                        pl_trainer_kwargs={"accelerator": "gpu", "gpus": -1, "auto_select_gpus": True} ) # Use all available GPUs
        else:
            cur_model_path = os.path.join(self.model_save_dir, "merged", "checkpoints")
            self.logger.info(f"loading merged model from {cur_model_path}")
            self.merged_model = self.model_builder.load_from_checkpoint("merged", work_dir=self.model_save_dir)


    def checkpoint_dir_exists(self):
        checkpoint_dir = os.path.join(self.model_save_dir)
        return os.path.exists(checkpoint_dir)


    def _load_existing_models(self):
        self.logger.info("Loading existing models.")
        models = []
        for index in range(self.dataset.num_X_sets):
            self.logger.info("Loading model for dataset %s" % index)

            cur_model_path = os.path.join(self.model_save_dir, str(index), "checkpoints")
            if os.path.exists(cur_model_path):
                self.logger.info("Checkpoints dir found. Loading model.")
                model = self.model_builder.load_from_checkpoint(str(index), work_dir=self.model_save_dir)
            else:
                self.logger.info("Checkpoints dir NOT found. No model will be loaded.")
                model = None
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
                                        batch_size=8,
                                        likelihood=self.likelihood,
                                        pl_trainer_kwargs={
                                            "accelerator": "gpu",
                                            "gpus": [0]
                                        })
            models.append(model)
        self.logger.info("All models built!")
        return models


    def fit_merged(self, epochs=20):
        self.logger.info("Fitting merged model")
        X_train = self.dataset.merged_X_train
        X_validation = self.dataset.merged_X_validation
        y_train = self.dataset.y_train
        y_val= self.dataset.y_validation
        self.merged_model.fit(series=y_train, past_covariates=X_train, 
                        val_series=y_val, val_past_covariates=X_validation, 
                        verbose=True, epochs=epochs)


    def fit(self, model_indexes=[0,1,2,3,4,5,6,7,8,9,10,11], epochs=20):
        """
        Wrapper for tf.keras.model.fit() to train internal model instance. Exposes select tuning parameters.

        Args:
            epochs (int, optional):  Number of epochs to train the model. Defaults to 20.
            batch_size (int, optional): Number of samples per gradient update. Defaults to 10.
            shuffle (bool, optional): Whether to shuffle the training data before each epoch. Defaults to True.
        """
        model_list = str(model_indexes)
        self.logger.info(f"Fitting models: {model_list}")

        models = self.models
        X_trains = self.dataset.X_trains
        X_validations = self.dataset.X_validations
        y_train = self.dataset.y_train
        y_val= self.dataset.y_validation

        for index in model_indexes:
            index = int(index)
            cur_model_path = os.path.join(self.model_save_dir, str(index), "checkpoints")
            if not os.path.exists(cur_model_path):
                self.logger.info(f"Model {index} not found. Building new model.")
                model = self.model_builder(**self.model_params, 
                                        work_dir=self.model_save_dir, model_name=str(index), 
                                        force_reset=True, save_checkpoints=True,
                                        batch_size=16,
                                        likelihood=self.likelihood,
                                        pl_trainer_kwargs={
                                            "accelerator": "gpu",
                                            "gpus": [0]
                                        })
            else:
                model = self.models[index]
            
            X_train = X_trains[index]
            X_val =X_validations[index]

            self.logger.info("Fitting model %s" % index)
            model.fit(series=y_train, past_covariates=X_train, 
                        val_series=y_val, val_past_covariates=X_val, 
                        verbose=True, epochs=epochs)


    def forecast_for_range(self, start, end):
        cur_timestep = start
        while cur_timestep < end:
            level = self.forecast_for(cur_timestep)
            self.prediction_set.record_level_for(cur_timestep, level)
            cur_timestep = cur_timestep + timedelta(hours=1)


    def historical_forecasts(self, **kwargs):
        """
        Create historical forecasts for the validation series using all models. Return a list of Timeseries
        """
        self.logger.info("Generating historical forecasts")
        y_preds_min = []
        y_preds_mid = []
        y_preds_max = []

        target_scaler = self.dataset.target_scaler
        y_test = self.dataset.y_test
        for index, (model, X_test) in enumerate(zip(self.models, self.dataset.X_tests)):
            self.logger.info("Generating historical forecast for model %s" % index)
            y_pred = model.historical_forecasts(series=y_test, past_covariates=X_test, start=0.5, retrain=False, overlap_end=False, last_points_only=True, verbose=True , **kwargs)
            y_pred = target_scaler.inverse_transform(y_pred)
            if y_pred.is_stochastic:
                self.logger.info("Current forecast identified as stochastic")
                y_pred_min = y_pred.quantile_df(0.05).applymap(lambda x: x.item())
                y_preds_min.append(y_pred_min)
                y_pred_mid = y_pred.quantile_df(0.5).applymap(lambda x: x.item())
                y_preds_mid.append(y_pred_mid)
                y_pred_max = y_pred.quantile_df(0.95).applymap(lambda x: x.item())
                y_preds_max.append(y_pred_max)
                df = self._join_preds(y_preds_min, y_preds_mid, y_preds_max)
            else:
                self.logger.info("Current forecast identified as deterministic")
                y_preds_mid.append(y_pred)
                df = self._join_preds(y_preds_mid, y_preds_mid, y_preds_mid)

        return df


    def merged_forecast_for_hours(self, n=24):
        """
        """
        self.logger.info("Generating future forecasts with merged model")

        target_scaler = self.dataset.target_scaler
        X = self.dataset.merged_X_current
        y = self.dataset.y_current
        
        y_pred = self.merged_model.predict(n=n, series=y, past_covariates=X)
        y_pred = target_scaler.inverse_transform(y_pred)

        if y_pred.is_stochastic:
            self.logger.info("Current forecast identified as stochastic")
            df = y_pred.quantiles_df()
        else:
            self.logger.info("Current forecast identified as deterministic")
            df = y_pred.pd_dataframe()

        return df


    def forecast_for_hours(self, n=24, num_samples=100):
        """
        """
        self.logger.info("Generating future forecasts")
        y_preds_min = []
        y_preds_mid = []
        y_preds_max = []

        target_scaler = self.dataset.target_scaler
        y = self.dataset.y_current
        for index, (model, X) in enumerate(zip(self.models, self.dataset.Xs_current)):
            self.logger.info("Generating future forecast for model %s" % index)
            y_pred = model.predict(n=n, series=y, past_covariates=X, num_samples=num_samples)
            y_pred = target_scaler.inverse_transform(y_pred)
            if y_pred.is_stochastic:
                self.logger.info("Current forecast identified as stochastic")
                y_pred_min = y_pred.quantile_df(0.05).applymap(lambda x: x.item())
                y_preds_min.append(y_pred_min)
                y_pred_mid = y_pred.quantile_df(0.5).applymap(lambda x: x.item())
                y_preds_mid.append(y_pred_mid)
                y_pred_max = y_pred.quantile_df(0.95).applymap(lambda x: x.item())
                y_preds_max.append(y_pred_max)
                df = self._join_preds(y_preds_min, y_preds_mid, y_preds_max)
            else:
                self.logger.info("Current forecast identified as deterministic")
                y_preds_mid.append(y_pred)
                df = self._join_preds(y_preds_mid, y_preds_mid, y_preds_mid)

        return df


    def _join_preds(self, y_preds_min, y_preds_mid, y_preds_max):
        df = pd.DataFrame()
        df['min'] = pd.concat(y_preds_min, axis=1).min(axis=1)
        df['mean'] = pd.concat(y_preds_mid, axis=1).mean(axis=1)
        df['max'] = pd.concat(y_preds_max, axis=1).max(axis=1)

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