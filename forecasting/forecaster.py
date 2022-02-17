import os
import numpy as np
from datetime import timedelta
from darts.models import BlockRNNModel

from forecasting.dataset import Dataset

class Forecaster:
    """
    Highest level user facing class. Each instance is specific to a single gauge site/catchment.
    Managers training data, inference data, model fitting and inference of trained model.
    Allows the user to query for specific forecasts or forecast ranges.
    """
    def __init__(self, catchment_data, model_builder=BlockRNNModel, overwrite_existing_models=False, checkpoint_dir="trained_models", verbose=True) -> None:
        """
        Fetches data from provided forecast site and generates processed training and inference sets.
        Builds the specified model.
        """
        self.verbose = verbose
        self.name = catchment_data.name
        self.dataset = Dataset(catchment_data)

        self.model_builder = model_builder
        self.model_save_dir = os.path.join(checkpoint_dir, self.name)
        os.makedirs(self.model_save_dir, exist_ok=True)

        self.models=[]
        if overwrite_existing_models:
            self.models = self._build_new_models()
        else:
            self.models = self._load_existing_models()
            

    def _load_existing_models(self):
        if self.verbose:
            print("Loading existing models")
        models = []
        for index in range(self.dataset.num_X_sets):
            if self.verbose:
                print("Loading model for set", index)
            model = self.model_builder.load_from_checkpoint(str(index), work_dir=self.model_save_dir)
            models.append(model)
        if self.verbose:
            print("All models loaded!")
        return models


    def _build_new_models(self):
        if self.verbose:
            print("Building new models. Overwritting any existing models.")
        models = []
        for index in range(self.dataset.num_X_sets):
            if self.verbose:
                print("Building model for set", index)
            model = self.model_builder(input_chunk_length=120, output_chunk_length=72, 
                                        work_dir=self.model_save_dir, model_name=str(index), 
                                        force_reset=True, save_checkpoints=True)
            models.append(model)
        if self.verbose:
            print("All models built!")
        return models

        
    def fit(self, epochs=1):
        """
        Wrapper for tf.keras.model.fit() to train internal model instance. Exposes select tuning parameters.

        Args:
            epochs (int, optional):  Number of epochs to train the model. Defaults to 20.
            batch_size (int, optional): Number of samples per gradient update. Defaults to 10.
            shuffle (bool, optional): Whether to shuffle the training data before each epoch. Defaults to True.
        """
        if self.verbose:
            print("Fitting all models")

        models = self.models
        X_trains = self.dataset.X_trains
        X_validations = self.dataset.X_validations
        y_train = self.dataset.y_train
        y_val= self.dataset.y_validation
        
        for index, (model, X_train, X_val) in enumerate(zip(models, X_trains, X_validations)):
            if self.verbose:
                print("Fitting model ", index)
            model.fit(series=y_train, past_covariates=X_train, 
                        val_series=y_val, val_past_covariates=X_val, 
                        verbose=True, epochs=epochs)


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


    def historical_forecasts(self):
        """
        Create historical forecasts for the validation series using all models. Return a list of Timeseries
        """
        if self.verbose:
            print("Generating historical forecasts")
        y_preds = []
        y_val = self.dataset.y_validation
        for index, (model, X_val) in enumerate(zip(self.models, self.dataset.X_validations)):
            if self.verbose:
                print("Generating historical forecast for model ", index)
            y_pred = model.historical_forecasts(series=y_val, past_covariates=X_val, num_samples=1, start=0.5, forecast_horizon=48, stride=24, retrain=False, overlap_end=False, last_points_only=True, verbose=True)
            y_preds.append(y_pred)
        return y_preds


    def update_input_data(self) -> None:
        """
        Force an update to ensure inference data is up to date. Run at least hourly when forecasting.
        """
        self.prediction_set.update()