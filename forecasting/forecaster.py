"""
This module provides the high level Forecaster class. 
"""
from forecasting.dataset import *

class Forecaster:
    """
    """
    def __init__(self, weather_locations, gauge_url, model_builder) -> None:
        """
        
        """
        self.historical_data = Dataset(weather_locations, gauge_url)
        self.model = model_builder(self.historical_data.input_shape)
        
        self.current_data = self.historical_data.X_test_shaped #InferenceSet(weather_locations)
        self.forecasted_levels = []
    def fit(self):
        self.model.fit(self.historical_data.X_train_shaped, self.historical_data.y_train, 
                        epochs = 20, 
                        batch_size = 10, 
                        shuffle = True)

    def forecast_for(self, index):
        """
        Take in the index for which a forecast is desired
        """
        x_in = self.current_data[index:index+1]
        y_pred = np.array(self.model.predict(x_in))
        target_scaler = self.historical_data.target_scaler
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