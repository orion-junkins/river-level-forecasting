#%%
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
        self.verbose = True
        self.weather_dataframes = []
        self.df_level = None
        self._process_all_weather_urls(weather_urls)
        self._process_level_url(level_url)


    def _process_level_url(self, level_url) -> None:
        """
        Fetch data from the given level url and perform basic processing.

        Args:
            level_url (string): exact url linking to the desired CSV file
        """
        self.df_level = pd.read_csv(level_url, sep='\t', comment='#') 

        cols_to_drop = [col for col in self.df_level.columns if 'cd' in col]

        cols_to_drop.append('site_no')

        self.df_level.drop(columns=cols_to_drop, inplace=True)
        self.df_level.drop(0, inplace=True)

        # Convert the datetime column to datetime objects
        self.df_level["datetime"] = pd.to_datetime(self.df_level["datetime"])
        
        # Use the datetime column as the index
        self.df_level.set_index('datetime', inplace=True)
        for col in self.df_level.columns:
            matched = re.match("[0-9]+_[0-9]+_*[0-9]*", col)
            is_match = bool(matched)
            if is_match:
                # Rename the level column
                self.df_level.rename(columns={col:'level'}, inplace=True)
        
        # Cast the level column to type float
        self.df_level['level'] = self.df_level['level'].astype(float)
        if self.verbose:
            print("Level data Fetched. Raw data following initial pre-pro:")
            display(self.df_level)


    def _process_all_weather_urls(self, weather_urls):
        """ 
        Fetch and process data from every given weather url in the list

        Args:
            weather_urls (list): List of exact urls linking to CSV files for the target weather stations.
        """
        for url in weather_urls:
            self._process_weather_url(url)
        if self.verbose:
            print("Weather data Fetched. Raw data following initial pre-pro:")
            for df_weather in self.weather_dataframes:
                display(df_weather)


    def _process_weather_url(self, url):
        """
        Fetch data from the given url, proccess it and add it to the list of weather datasets.

        Args:
            url (string): exact url linking to the target CSV
        """
        # Fetch data from url
        df_weather = pd.read_csv(url, comment='#') 
        

        # Drop un-needed metadata
        cols_to_drop = ['Precipitation Accumulation (in) Start of Day Values']
        for col in cols_to_drop:
            if col not in df_weather.columns:
                cols_to_drop.remove(col)

        df_weather.drop(columns=cols_to_drop, inplace=True)

        # Renamne the date column to match levels data
        df_weather.rename(columns={'Date':'datetime'}, inplace=True)

        # Convert the datetime column into datetime objects
        df_weather["datetime"] = pd.to_datetime(df_weather["datetime"])

        # Use the datetime column as the index
        df_weather.set_index('datetime', inplace=True)
        
        # Add the processed dataframe to the list
        self.weather_dataframes.append(df_weather)

    
def yesterday() -> str:
    """
    Helper function for data retrieval
    Returns:
        yesterday (string): yesterdays date in the format "%Y-%m-%d"
    """
    yesterday = datetime.today()  - timedelta(days=1)
    yesterday = yesterday.strftime("%Y-%m-%d")
    return yesterday

#%%
# Example usage: 
# Based on today's date, fetch all relevant water data
mck_vida_url = 'https://waterdata.usgs.gov/nwis/dv?cb_00060=on&format=rdb&site_no=14162500&referred_module=sw&period=&begin_date=1988-10-12&end_date=' + yesterday()

# Fetch weather & SWE data for Mckenzie basin. Atuomatically retrieves all data up to present.
mck1_weather_url ='https://wcc.sc.egov.usda.gov/reportGenerator/view_csv/customSingleStationReport/daily/619:OR:SNTL%7Cid=%22%22%7Cname/POR_BEGIN,POR_END/WTEQ::value,PREC::value,TMAX::value,TMIN::value,TAVG::value,PRCP::value'

mck2_weather_url = 'https://wcc.sc.egov.usda.gov/reportGenerator/view_csv/customMultiTimeSeriesGroupByStationReport/daily/start_of_period/719:OR:SNTL%7Cid=%22%22%7Cname/1980-11-03,2022-01-08/WTEQ::value,TMAX::value,TMIN::value,TAVG::value,PRCP::value?fitToScreen=false'

ds = Dataset([mck1_weather_url, mck2_weather_url], mck_vida_url)

# %%
