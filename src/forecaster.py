"""
This module provides the high level Forecaster class. 
"""
class Forecaster:
    """
    """
    def __init__(self, model) -> None:
        """
        
        """
        self.model = None # Fully trained inference model
        self.x_in = None # Up to date relevant input data
    

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