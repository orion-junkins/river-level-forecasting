import time
from datetime import datetime, timedelta


def convert_timestamp_to_datetime(timestamp: int | str, tz_offset: int = 0) -> str:
    """
    Convert a given Unix timestamp to a str datetime representation

    Args:
        timestamp (int or str): timestamp to convert
        tz_offset (int, optional): Timezone offset if desired. Defaults to 0.

    Returns:
        date (str): converted datetime
    """
    ts = int(timestamp) - tz_offset
    date = (datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S'))
    return date


def convert_timestamp_to_time(timestamp: int | str, tz_offset: int = 28800) -> str:
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
    yesterday = datetime.today() - timedelta(days=1)
    yesterday = yesterday.strftime("%Y-%m-%d")
    return yesterday


def date_days_ago(days: int) -> str:
    """
    Helper function for data retrieval. Gives string representation of date for given number of days in past

    Args:
        days (int): number of days in past for which date is desired
    Returns:
        date (str): date for present - days
    """
    date = datetime.now() - timedelta(days=days)
    date = date.strftime("%Y-%m-%d")
    return date


def unix_timestamp_days_ago(days: int) -> str:
    """
    Helper function for data retrieval. Gives unix timestamp of date for given number of days in past

    Args:
        days (int): number of days in past for which date is desired
    Returns:
        timestamp (str): unix timestamp for present - days
    """
    date = datetime.now() - timedelta(days=days)
    timestamp = str(int(time.mktime(date.timetuple())))
    return timestamp


def unix_timestamp_now() -> str:
    date = datetime.now()
    timestamp = str(int(time.mktime(date.timetuple())))
    return timestamp
