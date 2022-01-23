import json
import pandas as pd
import json
import urllib.request, json 
from datetime import datetime
import regex as re

from forecasting.data_fetching_utilities.open_weather_api_keys import api_key

# TODO investigate if any of these could be useful    
OPEN_WEATHER_DEFAULT_DROP_COLS = ['dt', 'dt_iso', 'timezone', 'city_name', 'lat', 'lon', 'visibility', 'sea_level', 'grnd_level', 'wind_gust', 'weather_description', 'weather_icon', 'clouds_all', 'weather_id', 'weather_main', 'rain_3h', 'snow_3h']
OPEN_WEATHER_COLUMNS = []

def get_forecast(lat, lon, excludes="current,minutely,hourly,alerts", api_key=api_key):
    request_url = f"https://api.openweathermap.org/data/2.5/onecall?lat={lat}&lon={lon}&exclude={excludes}&appid={api_key}"
    
    with urllib.request.urlopen(request_url) as url:
        data = json.loads(url.read().decode())
    df = pd.json_normalize(data, record_path =['daily'])
    return df


def get_forecasts(locations):
    forecasts = []
    for loc in locations:
        forecasts.append(get_forecast(loc[0], loc[1]))
    return forecasts


def get_single_loc_historical(path, drop_cols=OPEN_WEATHER_DEFAULT_DROP_COLS):
    df_weather = pd.read_csv(path)
    df_weather['datetime'] = list(map(datetime.fromtimestamp, df_weather['dt'])) 
    df_weather.set_index('datetime', inplace=True)

    # Filter out any columns that are present in the drop_cols list
    drop_cols = list(filter(lambda x: x in df_weather.columns, drop_cols))
    df_weather.drop(columns=drop_cols, inplace=True)

    # Convert Nan precip values to 0.0
    df_weather['rain_1h'].fillna(0.0, inplace=True)
    df_weather['snow_1h'].fillna(0.0, inplace=True)
    
    return df_weather


def get_all_historical_weather(paths, drop_cols=OPEN_WEATHER_DEFAULT_DROP_COLS):
    dfs_historical = []
    for path in paths:
        df_historical = get_single_loc_historical(path, drop_cols=drop_cols)
        dfs_historical.append(df_historical)
    
    return dfs_historical

def get_single_loc_recent(lat, lon, start, end, api_key=api_key):
    request_url = f"http://history.openweathermap.org/data/2.5/history/city?lat={lat}&lon={lon}&units=imperial&type=hour&start={start}&end={end}&appid={api_key}"
    print(request_url)
    with urllib.request.urlopen(request_url) as url:
        data = json.loads(url.read().decode())
    df_recent = pd.json_normalize(data, record_path =['list'])
    df_recent['datetime'] = list(map(datetime.fromtimestamp, df_recent['dt'])) 
    df_recent.set_index('datetime', inplace=True)
    df_recent = df_recent.rename(columns=lambda x: re.sub('main.','',x))
    return df_recent


def get_all_recent_weather(weather_locs, start):
    dfs_recent = []
    for loc in weather_locs:
        lat = loc[0]
        lon = loc[1]
        df_recent = get_single_loc_recent(lat, lon, start)
        dfs_recent.append(df_recent)
    return dfs_recent
