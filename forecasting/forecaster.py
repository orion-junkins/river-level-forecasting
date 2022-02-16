import numpy as np
from forecasting.dataset import Dataset
from datetime import datetime, timedelta
import os
import tensorflow as tf
from tensorflow import keras

class Forecaster:
    """
    Highest level user facing class. Each instance is specific to a single gauge site/catchment.
    Managers training data, inference data, model fitting and inference of trained model.
    Allows the user to query for specific forecasts or forecast ranges.
    """
    def __init__(self, catchment_data, model_builder, checkpoint_dir="training/") -> None:
        """
        Fetches data from provided forecast site and generates processed training and inference sets.
        Builds the specified model.
        """
        self.dataset = Dataset(catchment_data)
        self.models = [model_builder() for X_test in self.dataset.X_tests if X_test != None]
        self.checkpoint_path = os.path.join(checkpoint_dir, "cp.ckpt")

    def fit(self, epochs=2):
        """
        Wrapper for tf.keras.model.fit() to train internal model instance. Exposes select tuning parameters.

        Args:
            epochs (int, optional):  Number of epochs to train the model. Defaults to 20.
            batch_size (int, optional): Number of samples per gradient update. Defaults to 10.
            shuffle (bool, optional): Whether to shuffle the training data before each epoch. Defaults to True.
        """
        print("Fitting Models")

        models = self.models
        X_trains = self.dataset.X_trains
        X_validations = self.dataset.X_validations
        y_train = self.dataset.y_train
        y_val= self.dataset.y_validation
        for index, (model, X_train, X_val) in enumerate(zip(models, X_trains, X_validations)):
            print("Fitting model ", index)
            model.fit(series=y_train, past_covariates=X_train, 
                        val_series=y_val, val_past_covariates=X_val, 
                        verbose=True, epochs=epochs)


    def load_trained(self):
        print("Loading trained model")
        self.model.load_weights(self.checkpoint_path)


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


    def update_input_data(self) -> None:
        """
        Force an update to ensure inference data is up to date. Run at least hourly when forecasting.
        """
        self.prediction_set.update()