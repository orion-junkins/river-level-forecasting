#%%
import json
import pandas as pd
import json
import urllib.request, json 
from datetime import datetime

from ...open_weather_api_keys import api_key

def fetch_forecast(lat, lon, start=None, excludes="current,minutely,hourly,alerts", api_key=api_key):
    request_url = f"https://api.openweathermap.org/data/2.5/onecall?lat={lat}&lon={lon}&exclude={excludes}&appid={api_key}"
    
    with urllib.request.urlopen(request_url) as url:
        data = json.loads(url.read().decode())
        print(data)
    df = pd.json_normalize(data, record_path =['daily'])
    return df

df = fetch_forecast(44.21, -121.87)
df_backup = df
#%%
df = df_backup

def convert_timestamp_to_datetime(timestamp):
    ts = int(timestamp) - 28800
    return (datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S'))

def convert_timestamp_to_time(timestamp):
    ts = int(timestamp) - 28800
    return (datetime.utcfromtimestamp(ts).strftime('%H:%M:%S'))

df['dt'] = df['dt'].apply(convert_timestamp_to_datetime)
df['sunrise'] = df['sunrise'].apply(convert_timestamp_to_time)
df['sunset'] = df['sunset'].apply(convert_timestamp_to_time)
