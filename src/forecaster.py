"""
This module provides the high level Forecaster class. 
"""
from dataset import *

class Forecaster:
    """
    """
    def __init__(self, weather_locations, gauge_url, model) -> None:
        """
        
        """
        self.historical_data = Dataset(weather_locations, gauge_url)
        self.model = model.fit(self.data.X_train_shaped, self.data.y_train, 
                        epochs = 20, 
                        batch_size = 10, 
                        shuffle = True)
    

    @property
    def forecast(self):
        """
        """
        self.model.run_inference(self.x_in)


    def forecast_for(self, timestamp) -> str:
        """
        """
        expected_level = self.forecast[timestamp]
        
        return "Expected level for " + timestamp + " is " + expected_level


    def update_input_data(self, input_data) -> None:
        """
        """
        self.input_data = input_data
        