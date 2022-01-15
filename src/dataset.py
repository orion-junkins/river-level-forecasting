"""
This module provides data set processing utilities. Core functionality lies within the 'Dataset' class.
Other helper functions are defined below. 
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import re
from IPython.display import display

class Dataset:
    """ 
    Wrapper class for combined weather and level data sets. 
    """
    def __init__(self, weather_urls, level_url) -> None:
        self.verbose = True
        self.scaler = MinMaxScaler()
        self.target_scaler = MinMaxScaler()

        self.weather_dataframes = self._process_all_weather_urls(weather_urls)
        self.df_level = self._process_level_url(level_url)
        self.df_merged = self._merge_all()
        self.df_processed = self._process_merged()
        self.X, self.y = self._build_X_y()
        self.X_train, self.X_test, self.y_train, self.y_test = self._partition()


    def _process_level_url(self, level_url) -> None:
        """
        Fetch data from the given level url and perform basic processing.

        Args:
            level_url (string): exact url linking to the desired CSV file
        """
        df_level = pd.read_csv(level_url, sep='\t', comment='#') 

        cols_to_drop = [col for col in df_level.columns if 'cd' in col]

        cols_to_drop.append('site_no')

        df_level.drop(columns=cols_to_drop, inplace=True)
        df_level.drop(0, inplace=True)

        # Convert the datetime column to datetime objects
        df_level["datetime"] = pd.to_datetime(df_level["datetime"])
        
        # Use the datetime column as the index
        df_level.set_index('datetime', inplace=True)
        for col in df_level.columns:
            matched = re.match("[0-9]+_[0-9]+_*[0-9]*", col)
            is_match = bool(matched)
            if is_match:
                # Rename the level column
                df_level.rename(columns={col:'level'}, inplace=True)
        
        # Cast the level column to type float
        df_level['level'] = df_level['level'].astype(float)
        if self.verbose:
            print("Level data Fetched. Raw data following initial pre-pro:")
            display(df_level)

        return df_level


    def _process_all_weather_urls(self, weather_urls):
        """ 
        Fetch and process data from every given weather url in the list

        Args:
            weather_urls (list): List of exact urls linking to CSV files for the target weather stations.
        """
        weather_dataframes = []
        for url in weather_urls:
            weather_dataframes.append(self._process_weather_url(url))

        if self.verbose:
            print("Weather data Fetched. Raw data following initial pre-pro:")
            for df_weather in weather_dataframes:
                display(df_weather)

        return weather_dataframes


    def _process_weather_url(self, url):
        """
        Fetch data from the given url, process it and add it to the list of weather datasets.

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
        
        return(df_weather)


    def _merge_all(self):
        temp_weather_dataframes = self.weather_dataframes
        df_merged = temp_weather_dataframes.pop(0)

        for df_weather in temp_weather_dataframes:
            df_merged = pd.merge(df_merged, df_weather, on="datetime")
        
        df_merged = pd.merge(df_merged, self.df_level, on="datetime")

        if self.verbose:
            print("Data merged. Full data frame following merge:")
            display(df_merged)
        
        return df_merged


    def _process_merged(self):
        df_processed = self.df_merged

        df_processed['next_level'] = np.nan
        rows = df_processed.shape[0]
        for row_idx in range(0, rows-1):
            df_processed['next_level'][row_idx] = df_processed['level'][row_idx+1]

        # Impute NaNs by averaging backfilled and forward filled approachess
        # Essentially, this will average nearest non NaN neighbors on either side sequentially
        # Compute forward/back filled data
        for_fill = df_processed.fillna(method='ffill')
        back_fill = df_processed.fillna(method='bfill')

        # For every column in the dataframe,
        for col in df_processed.columns:
            # Average the forward and back filled values
            df_processed[col] = (for_fill[col] + back_fill[col])/2

        # TODO: Move all row drops past sequencing
        # Drop any rows remaining which have NaN values (generally first and/or last rows)
        df_processed.dropna(inplace=True)

        # Confirm imputation worked
        assert(df_processed.isna().sum().sum() == 0)

        # For every feature column,
        for column in df_processed.columns[:-1]:
            # fit and transform the data
            df_processed[[column]] = self.scaler.fit_transform(df_processed[[column]])

        # Scale the target column
        target_col = df_processed.columns[-1]
        df_processed[[target_col]] = self.target_scaler.fit_transform(df_processed[[target_col]])

        # Display the newly scaled dataframe
        if self.verbose: 
            display(df_processed)
        
        return df_processed


    def _build_X_y(self, window_length=5):
        X = self.df_processed.iloc[:,:-1].values
        y = self.df_processed.iloc[:,-1].values

        num_samples = len(X) - window_length

        windowed_X = []
        windowed_y = []
        for index in range(num_samples):
            current_window_end = index + window_length
            cur_X_seq = X[index:current_window_end, :]
            windowed_X.append(cur_X_seq)

            windowed_y.append(y[current_window_end])

        X = np.array(windowed_X)
        y = np.array(windowed_y)
        
        return (X, y)


    def _partition(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size = 0.2, random_state = 0)
        return (X_train, X_test, y_train, y_test)