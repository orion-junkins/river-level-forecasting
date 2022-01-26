import json
import pandas as pd
import json
import urllib.request, json 
from datetime import datetime
import regex as re

from forecasting.data_fetching_utilities.open_weather_api_keys import api_key
from forecasting.time_utils import *

# TODO explore other features beyond this set
DEFAULT_WEATHER_COLS = ['temp','pressure', 'humidity', 'wind_speed', 'wind_deg', 'rain_1h', 'snow_1h']

# Open Weather API wrapper functions
def fetch_hourly_forecast(lat, lon, api_key=api_key):
    request_url = f"http://pro.openweathermap.org/data/2.5/forecast/hourly?lat={lat}&lon={lon}&appid={api_key}"
    df = fetch_to_dataframe(request_url, ['list'])
    df = correct_columns(df)
    return df


def fetch_recent_historical(lat, lon, start, end=unix_timestamp_now(), api_key=api_key):
    request_url = f"http://history.openweathermap.org/data/2.5/history/city?lat={lat}&lon={lon}&units=imperial&type=hour&start={start}&end={end}&appid={api_key}"
    print(request_url)
    df = fetch_to_dataframe(request_url, ['list'])
    df = correct_columns(df)
    return df


def load_single_loc_historical(path):
    df = pd.read_csv(path)
    df['datetime'] = list(map(datetime.fromtimestamp, df['dt'])) 
    df.set_index('datetime', inplace=True)

    df = correct_columns(df)

    # Convert Nan precip values to 0.0
    df['rain_1h'].fillna(0.0, inplace=True)
    df['snow_1h'].fillna(0.0, inplace=True)
    return df



# Wrappers to query multiple locations at once [] -> []
def get_all_forecasted_weather(locations) -> list:
    forecasts = []
    for loc in locations:
        lat = loc[0]
        lon = loc[1]
        forecasts.append(fetch_hourly_forecast(lat, lon))
    return forecasts


def get_all_recent_weather(weather_locs, start) -> list:
    dfs_recent = []
    for loc in weather_locs:
        lat = loc[0]
        lon = loc[1]
        df_recent = fetch_recent_historical(lat, lon, start)
        dfs_recent.append(df_recent)
    return dfs_recent

    
def get_all_historical_weather(paths) -> list:
    dfs_historical = []
    for path in paths:
        df_historical = load_single_loc_historical(path)
        dfs_historical.append(df_historical)
    
    return dfs_historical



# Helper functions
def fetch_to_dataframe(request_url, record_path):
    with urllib.request.urlopen(request_url) as url:
        data = json.loads(url.read().decode())
    df = pd.json_normalize(data, record_path =record_path)
    df['datetime'] = list(map(datetime.fromtimestamp, df['dt'])) 
    df.set_index('datetime', inplace=True)
    return df


def correct_columns(df, target_cols=DEFAULT_WEATHER_COLS):
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

