"""
This module provides data set processing utilities. Core functionality lies within the 'Dataset' class.
Other helper functions are defined below. 
"""
import pandas as pd
from datetime import datetime, timedelta
import re
from IPython.display import display

class Dataset:
    """ 
    Wrapper class for combined weather and level data sets. 
    """
    def __init__(self, weather_urls, level_url) -> None:
        self.weather_data = []
        self.level_data = None
        #self._process_all_weather_urls(weather_urls)
        self._process_level_url(level_url)


    def _process_level_url(self, level_url) -> None:
        """
        Fetch data from the given level url and perform basic processing.

        Args:
            level_url (string): exact url linking to the desired CSV file
        """
        self.level_data = pd.read_csv(level_url, sep='\t', comment='#') 

        cols_to_drop = [col for col in self.level_data.columns if 'cd' in col]

        cols_to_drop.append('site_no')

        self.level_data.drop(columns=cols_to_drop, inplace=True)
        self.level_data.drop(0, inplace=True)

        # Convert the datetime column to datetime objects
        self.level_data["datetime"] = pd.to_datetime(self.level_data["datetime"])
        
        # Use the datetime column as the index
        self.level_data.set_index('datetime', inplace=True)
        for col in self.level_data.columns:
            matched = re.match("[0-9]+_[0-9]+_*[0-9]*", col)
            is_match = bool(matched)
            if is_match:
                # Rename the level column
                self.level_data.rename(columns={col:'level'}, inplace=True)
        
        # Cast the level column to type float
        self.level_data['level'] = self.level_data['level'].astype(float)
        print("Level data Fetched. Raw data following initial pre-pro:")
        display(self.level_data)


    def _process_all_weather_urls(self, weather_urls):
        """ 
        Fetch and process data from every given weather url in the list

        Args:
            weather_urls (list): List of exact urls linking to CSV files for the target weather stations.
        """
        for url in weather_urls:
            self._process_weather_url(url)

    def _process_weather_url(self, url):
        """
        Fetch data from the given url, proccess it and add it to the list of weather datasets.

        Args:
            url (string): exact url linking to the target CSV
        """
        weather_data = pd.read_csv(url, comment='#')  

    
def yesterday() -> str:
    """
    Helper function for data retrieval
    Returns:
        yesterday (string): yesterdays date in the format "%Y-%m-%d"
    """
    yesterday = datetime.today()  - timedelta(days=1)
    yesterday = yesterday.strftime("%Y-%m-%d")
    return yesterday


# Example usage: 
# Based on today's date, fetch all relevant water data
mck_vida_url = 'https://waterdata.usgs.gov/nwis/dv?cb_00060=on&format=rdb&site_no=14162500&referred_module=sw&period=&begin_date=1988-10-12&end_date=' + yesterday()

ds = Dataset("", mck_vida_url)