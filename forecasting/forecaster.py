"""
This module provides the high level Forecaster class. 
"""
import numpy as np
from forecasting.training_set import TrainingSet
from forecasting.inference_set import InferenceSet

class Forecaster:
    """
    """
    def __init__(self, forecast_site, model_builder) -> None:
        """
        
        """
        self.training_set = TrainingSet(forecast_site)
        self.inference_set = InferenceSet(forecast_site)
        self.model = model_builder(self.training_set.input_shape)
        
        self.forecasted_levels = []


    def fit(self, epochs=20, batch_size=10, shuffle=True):
        """"""
        self.model.fit(self.training_set.X_train_shaped, self.training_set.y_train, 
                        epochs = epochs, 
                        batch_size = batch_size, 
                        shuffle = shuffle)

    def forecast_for(self, index):
        """
        Take in the index for which a forecast is desired
        """
        x_in = self.current_data[index:index+1]
        y_pred = np.array(self.model.predict(x_in))
        target_scaler = self.training_set.target_scaler
        y_pred = target_scaler.inverse_transform(y_pred)
        return y_pred[0][0]  


    def forecast_all(self):
        """
        """
        for i in range(len(self.current_data)):
            expected_level = self.forecast_for(i)
            self.forecasted_levels.append(expected_level)


    def update_input_data(self, input_data) -> None:
        """
        """
        self.input_data = input_data