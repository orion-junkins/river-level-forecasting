from datetime import datetime, timedelta

def convert_timestamp_to_datetime(timestamp, tz_offset=28800):
    """
    Convert a given Unix timestamp to a str datetime representation

    Args:
        timestamp (int or str): timestamp to convert
        tz_offset (int, optional): Timezone offset if desired. Defaults to 28800 (converts OpenWeather default to PST).

    Returns:
        date (str): converted datetime
    """
    ts = int(timestamp) - tz_offset
    date = (datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S'))
    return date 


def convert_timestamp_to_time(timestamp, tz_offset=28800):
    """
    Convert a given Unix timestamp to a str representation of time only, dropping date data

    Args:
        timestamp (int or str): timestamp to convert
        tz_offset (int, optional): Timezone offset if desired. Defaults to 28800 (converts OpenWeather default to PST).

    Returns:
        time (str): converted time
    """
    ts = int(timestamp) - tz_offset
    time = (datetime.utcfromtimestamp(ts).strftime('%H:%M:%S'))
    return time


def yesterday() -> str:
    """
    Helper function for data retrieval. Gives yesterdays date as a str representation of datetime
    
    Returns:
        yesterday (string): yesterdays date in the format "%Y-%m-%d"
    """
    yesterday = datetime.today()  - timedelta(days=1)
    yesterday = yesterday.strftime("%Y-%m-%d")
    return yesterday