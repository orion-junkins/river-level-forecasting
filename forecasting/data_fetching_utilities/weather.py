"""
Utility functions for fetching weather from the OpenWeatherMap API.
Note that historical weather more than 1 year ago cannot currently be fetched through API queries. Use given helper to load CSV files accordingly.
"""
import json
import pandas as pd
from pandas import DataFrame
import json
import urllib.request, json 
from datetime import datetime
import regex as re

from forecasting.data_fetching_utilities.open_weather_api_keys import api_key
from forecasting.general_utilities.time_utils import *

DEFAULT_WEATHER_COLS = ['temp','pressure', 'humidity', 'wind_speed', 'wind_deg', 'rain_1h', 'snow_1h']

# Open Weather API wrapper functions
def fetch_hourly_forecast(lat, lon, api_key=api_key) -> DataFrame:
    """
    Access an hourly forecast for the next 96 hours.

    Args:
        lat (str): latitude for which weather is desired.
        lon (str): longitude for which weather is desired.
        api_key (str, optional): OpenWeatherMap API key. Update var in open_weather_api_key.py. Defaults to api_key.

    Returns:
        df (dataframe): Fetched dataframe of forecast
    """
    request_url = f"http://pro.openweathermap.org/data/2.5/forecast/hourly?lat={lat}&lon={lon}&appid={api_key}"
    df = fetch_to_dataframe(request_url, ['list'])
    df = correct_columns(df)
    df = handle_missing_data(df)
    return df


def fetch_recent_historical(lat, lon, start, end=unix_timestamp_now(), api_key=api_key) -> DataFrame:
    """
    Access hourly historical weather data for the given location and time range.

    Args:
        lat (str): latitude for which weather is desired.
        lon (str): longitude for which weather is desired.
        start (str): unix timestamp for the start of the window for which weather is desired.
        end (str): unix timestamp for the end of the window for which weather is desired. Defaults to now.
        api_key (str, optional): OpenWeatherMap API key. Update var in open_weather_api_key.py. Defaults to api_key.

    Returns:
        df (DataFrame): Fetched dataframe of forecast
    """
    request_url = f"http://history.openweathermap.org/data/2.5/history/city?lat={lat}&lon={lon}&units=imperial&type=hour&start={start}&end={end}&appid={api_key}"
    df = fetch_to_dataframe(request_url, ['list'])
    df = correct_columns(df)

    df = handle_missing_data(df)
    return df


def load_single_loc_historical(path) -> DataFrame:
    """
    Load a file of historical data.

    Args:
        path (str): path to csv file.

    Returns:
        df (DataFrame): Fetched dataframe of weather.
    """
    df = pd.read_csv(path)
    df['datetime'] = list(map(datetime.fromtimestamp, df['dt'])) 
    df.set_index('datetime', inplace=True)
    df.index = pd.DatetimeIndex(df.index)
    df.index = df.index.tz_localize('utc')
    df.index = df.index.tz_convert(None)

    df = correct_columns(df)

    # Convert Nan precip values to 0.0
    df['rain_1h'].fillna(0.0, inplace=True)
    df['snow_1h'].fillna(0.0, inplace=True)

    df = handle_missing_data(df)
    return df



# Wrappers to query multiple locations at once [] -> []
def get_all_forecasted_weather(locations) -> list:
    """
    Fetch forecasted weather for multiple locations.

    Args:
        locations (list): list of ('lat', 'lon') tuples.

    Returns:
        list: list of DataFrame objects, one per location in initial list.
    """
    forecasts = []
    for loc in locations:
        lat = loc[0]
        lon = loc[1]
        forecasts.append(fetch_hourly_forecast(lat, lon))
    return forecasts


def get_all_recent_weather(locations, start) -> list:
    """
    Get recent weather data for all locations up to present.

    Args:
        locations (list): list of ('lat', 'lon') tuples.
        start (str): unix timestamp for the start of the window for which weather is desired.

    Returns:
        list: list of DataFrame objects, one per location in initial list.
    """
    dfs_recent = []
    for loc in locations:
        lat = loc[0]
        lon = loc[1]
        df_recent = fetch_recent_historical(lat, lon, start)
        dfs_recent.append(df_recent)
    return dfs_recent

    
def get_all_historical_weather(paths) -> list:
    """
    Load a file of historical data for each path in the given list.

    Args:
        paths (list): paths to csv files.

    Returns:
        list: list of DataFrame objects, one per path in initial list.
    """
    dfs_historical = []
    for path in paths:
        df_historical = load_single_loc_historical(path)
        dfs_historical.append(df_historical)
    
    return dfs_historical



# Helper functions
def fetch_to_dataframe(request_url, record_path) -> DataFrame:
    """
    Fetch data from the json file at the given url and process it into a dataframe.

    Args:
        request_url (str): path to file.
        record_path (str): path within json file to desired data.

    Returns:
        df (DataFrame): processed dataframe.
    """
    with urllib.request.urlopen(request_url) as url:
        data = json.loads(url.read().decode())
    df = pd.json_normalize(data, record_path =record_path)
    df['datetime'] = list(map(datetime.fromtimestamp, df['dt'])) 
    df.set_index('datetime', inplace=True)
    df.index = pd.DatetimeIndex(df.index)
    df.index = df.index.tz_localize('utc')
    df.index = df.index.tz_convert(None)

    return df


def correct_columns(df, target_cols=DEFAULT_WEATHER_COLS):
    """
    Modify the given df so that columns align with the given list. Rename columns in select cases, remove excess columns, and add columns of Nan/0 where appropriate

    Args:
        df (DataFrame): dataframe to be altered
        target_cols (list, optional): List of desired column names. Defaults to DEFAULT_WEATHER_COLS.

    Returns:
        df (DataFrame): corrected dataframe.
    """
    # Rename any cols with 'main.' prefix
    df = df.rename(columns=lambda x: re.sub('main.','',x))

    # Replace any remaining '.' chars in col names to '_'
    df = df.rename(columns=lambda x: re.sub('\.','_',x))

    # Filter out any columns that are present in the drop_cols list
    drop_cols = list(filter(lambda x: x not in target_cols, df.columns))
    df.drop(columns=drop_cols, inplace=True)

    # Create columns of 0 for any missing (not observed) data
    missing_cols = list(filter(lambda x: x not in df.columns, target_cols))
    for col in missing_cols:
        df[col] = 0

    return df

def handle_missing_data(df):
    if df.index[0].minute != 0:
        df.drop([df.index[0]], inplace=True)
    assert(df.index[0].minute == 0)

    # Remove duplicated entries
    df = df.loc[~df.index.duplicated(), :]

    # Set frequency as hourly
    df = df.asfreq('H')

    # Compute forward/back filled data
    for_fill = df.fillna(method='ffill')
    back_fill = df.fillna(method='bfill')
    # For every column in the dataframe,
    for col in df.columns:
        # Average the forward and back filled values
        df[col] = (for_fill[col] + back_fill[col])/2

    # Drop any rows remaining which have NaN values (generally first and/or last rows)
    df.dropna(inplace=True)

    # Confirm imputation worked
    assert(df.isna().sum().sum() == 0)

    return df