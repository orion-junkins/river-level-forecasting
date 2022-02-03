import numpy as np
from forecasting.dataset import Dataset
from forecasting.prediction_set import PredictionSet
import os
import tensorflow as tf
from tensorflow import keras

class Forecaster:
    """
    Highest level user facing class. Each instance is specific to a single gauge site/catchment.
    Managers training data, inference data, model fitting and inference of trained model.
    Allows the user to query for specific forecasts or forecast ranges.
    """
    def __init__(self, forecast_site, model_builder, checkpoint_dir="training/") -> None:
        """
        Fetches data from provided forecast site and generates processed training and inference sets.
        Builds the specified model.
        """
        print("Building dataset")
        self.dataset = Dataset(forecast_site)
        print("Building prediction set")
        self.prediction_set = PredictionSet(forecast_site, self.dataset.input_shape, self.dataset.scaler)
        print("Building Model")
        self.model = model_builder(self.dataset.input_shape)
        self.checkpoint_path = os.path.join(checkpoint_dir, "cp.ckpt")

    def fit(self, epochs=20, batch_size=10, shuffle=True):
        """
        Wrapper for tf.keras.model.fit() to train internal model instance. Exposes select tuning parameters.

        Args:
            epochs (int, optional):  Number of epochs to train the model. Defaults to 20.
            batch_size (int, optional): Number of samples per gradient update. Defaults to 10.
            shuffle (bool, optional): Whether to shuffle the training data before each epoch. Defaults to True.
        """
        print("Fitting Model")

        # Create a callback that saves the model's weights
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=self.checkpoint_path,
                                                        save_weights_only=True,
                                                        verbose=1)

        self.model.fit(self.dataset.X_train_shaped, self.dataset.y_train, 
                        epochs = epochs, 
                        batch_size = batch_size, 
                        shuffle = shuffle,
                        callbacks=[cp_callback])

    def load_trained(self):
        print("Loading trained model")
        self.model.load_weights(self.checkpoint_path)

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