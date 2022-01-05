"""
This module provides the high level Forecaster class. 
"""
class Forecaster:
    """
    """
    def __init__(self) -> None:
        """
        
        """
        self.model = None # Fully trained inference model
        self.input_data = None # Up to date relevant input data
    

    def forecast_for(timestamp) -> str:
        """
        """
        return "Expected change in level for " + timestamp + " is..."


    def update_input_data(self, input_data) -> None:
        """
        """
        self.input_data = input_data