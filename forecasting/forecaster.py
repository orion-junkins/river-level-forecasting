import numpy as np
from forecasting.training_set import TrainingSet
from forecasting.inference_set import InferenceSet

class Forecaster:
    """
    Highest level user facing class. Each instance is specific to a single gauge site/catchment.
    Managers training data, inference data, model fitting and inference of trained model.
    Allows the user to query for specific forecasts or forecast ranges.
    """
    def __init__(self, forecast_site, model_builder) -> None:
        """
        Fetches data from provided forecast site and generates processed training and inference sets.
        Builds the specified model.
        """
        self.training_set = TrainingSet(forecast_site)
        self.inference_set = InferenceSet(forecast_site, self.training_set.input_shape, self.training_set.scaler)
        self.model = model_builder(self.training_set.input_shape)


    def fit(self, epochs=20, batch_size=10, shuffle=True):
        """
        Wrapper for tf.keras.model.fit() to train internal model instance. Exposes select tuning parameters.

        Args:
            epochs (int, optional):  Number of epochs to train the model. Defaults to 20.
            batch_size (int, optional): Number of samples per gradient update. Defaults to 10.
            shuffle (bool, optional): Whether to shuffle the training data before each epoch. Defaults to True.
        """
        self.model.fit(self.training_set.X_train_shaped, self.training_set.y_train, 
                        epochs = epochs, 
                        batch_size = batch_size, 
                        shuffle = shuffle)


    def forecast_for(self, timestamp):
        """
        Run inference for the given timestamp.

        Args:
            timestamp (datetime): The date & time for which a forecast is desired. Must be a member of InferenceSet indices.

        Returns:
            y_pred (float): The predicted river level at the given timestamp
        """
        x_in = self.inference_set.x_in_for_window(timestamp)
        x_in = np.array([x_in])
        y_pred = np.array(self.model.predict(x_in))

        # Inverse transform result
        target_scaler = self.training_set.target_scaler
        y_pred = target_scaler.inverse_transform(y_pred)

        # Convert to float from np.array
        y_pred = y_pred[0][0]  
        return y_pred


    def update_input_data(self) -> None:
        """
        Force an update to ensure inference data is up to date. Run at least hourly when forecasting.
        """
        self.inference_set.update()