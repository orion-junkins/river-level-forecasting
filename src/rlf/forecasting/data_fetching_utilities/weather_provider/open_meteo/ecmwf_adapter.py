from typing import List, Optional
from rlf.forecasting.data_fetching_utilities.coordinate import Coordinate
from rlf.forecasting.data_fetching_utilities.weather_provider.api.base_api_adapter import BaseAPIAdapter
from rlf.forecasting.data_fetching_utilities.weather_provider.open_meteo.parameters import get_hourly_parameters
import openmeteo_requests
from retry_requests import retry
import requests_cache
from openmeteo_sdk import WeatherApiResponse


class OpenMeteoECMWFAdapter(BaseAPIAdapter):
    """Adapts the OpenMeteo API to be used by the RequestBuilder"""

    def __init__(self, archive_hourly_parameters: Optional[List[str]] = None,
                 forecast_hourly_parameters: Optional[List[str]] = None) -> None:
        """
        Adapts the OpenMeteo API to be used by the RequestBuilder

        Args:
            archive_hourly_parameters (list[str], optional): Which parameters to fetch for archived/historical queries. Defaults to None.
            forecast_hourly_parameters (list[str], optional): Which parameters to fetch for current/forecasted queries. Defaults to None.
        """
        self.archive_hourly_parameters = archive_hourly_parameters if archive_hourly_parameters is not None else get_hourly_parameters("ecmwf_shared")
        self.forecast_hourly_parameters = forecast_hourly_parameters if forecast_hourly_parameters is not None else get_hourly_parameters("ecmwf_shared")

    def get_historical(self,
                       coordinate: Coordinate,
                       start_date: str,
                       end_date: str,
                       columns: Optional[List[str]] = None) -> WeatherApiResponse:
        """Make a GET request to the Open Meteo API for historical/archived data.

        Args:
            coordinate (Coordinate): The location to fetch data for.
            start_date (str): The starting date for the requested data. In the format "YYYY-MM-DD".
            end_date (str): The ending date for the requested data. In the format "YYYY-MM-DD".
            columns (list[str], optional): The subset of columns to fetch. If set to None, all columns will be fetched. Defaults to None.

        Returns:
            response: The WeatherApiResponse object from open meteo, containing Hourly Variables, Longitude, Latitude, etc.
        """
        # Setup the Open-Meteo API client with cache and retry on error
        cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
        retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
        openmeteo = openmeteo_requests.Client(session=retry_session)

        hourly_params = columns if columns is not None else self.archive_hourly_parameters
        # Make sure all required weather variables are listed here
        # The order of variables in hourly or daily is important to assign them correctly below
        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": coordinate.lat,
            "longitude": coordinate.lon,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": hourly_params,
            "timezone": "GMT",
            "models": "ecmwf_ifs"
        }

        return openmeteo.weather_api(url, params=params)[0]

    def get_current(self,
                    coordinate: Coordinate,
                    past_days: int = 92,
                    forecast_days: int = 16,
                    columns: Optional[List[str]] = None) -> WeatherApiResponse:
        """Make a GET request to the Open Meteo API for current/forecasted data.

        Args:
            coordinate (Coordinate): The location to fetch data for.
            past_days (int, optional): How many days into the past to fetch data for. Defaults to 92 (OpenMeteo max value).
            forecast_days (int, optional): How many days into the future to fetch data for. Defaults to 16 (OpenMeteo max value).
            columns (list[str], optional): The subset of columns to fetch. If set to None, all columns will be fetched. Defaults to None.

        Returns:
            response: The WeatherApiResponse object from open meteo, containing Hourly Variables, Longitude, Latitude, etc.
        """
        # Setup the Open-Meteo API client with cache and retry on error
        cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
        retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
        openmeteo = openmeteo_requests.Client(session=retry_session)

        hourly_params = columns if columns is not None else self.forecast_hourly_parameters

        url = "https://api.open-meteo.com/v1/ecmwf"
        params = {
            "latitude": coordinate.lat,
            "longitude": coordinate.lon,
            "hourly": hourly_params,
            "past_days": past_days,
            "forecast_days": forecast_days
        }
        return openmeteo.weather_api(url, params=params)[0]

    def get_index_parameter(self) -> str:
        """Temporal index parameter for OpenMeteo hourly data is "time".

        Returns:
            str: "time"
        """
        return "time"
