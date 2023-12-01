import sys
print(sys.path)

from typing import List, Optional

from rlf.forecasting.data_fetching_utilities.coordinate import Coordinate
from rlf.forecasting.data_fetching_utilities.weather_provider.api.base_api_adapter import BaseAPIAdapter
from rlf.forecasting.data_fetching_utilities.weather_provider.api.models import Response
from rlf.forecasting.data_fetching_utilities.weather_provider.api.rest_invoker import RestInvoker
from rlf.forecasting.data_fetching_utilities.weather_provider.open_meteo.parameters import get_hourly_parameters

import openmeteo_requests
from retry_requests import retry
import requests_cache


class OpenMeteoECMWFAdapter(BaseAPIAdapter):
    """Adapts the OpenMeteo API to be used by the RequestBuilder"""

    def __init__(self, archive_hourly_parameters: Optional[List[str]] = None,
                 forecast_hourly_parameters: Optional[List[str]] = None) -> None:
        """
        Adapts the OpenMeteo API to be used by the RequestBuilder

        Args:
            protocol (str, optional): The protocol to use. Defaults to "https".
            archive_hostname (str, optional): The hostname to use for archived/historical data. Defaults to "archive-api.open-meteo.com".
            forecast_hostname (str, optional): The hostname to use for current/forecasted data. Defaults to "api.open-meteo.com".
            version (str, optional): The version of the API to use. Defaults to "v1".
            archive_path (str, optional): The path to use for archived/historical data. Defaults to "era5".
            forecast_path (str, optional): The path to use for current/forecasted data. Defaults to "gfs".
            archive_hourly_parameters (list[str], optional): Which parameters to fetch for archived/historical queries. Defaults to None.
            forecast_hourly_parameters (list[str], optional): Which parameters to fetch for current/forecasted queries. Defaults to None.
        """
        self.archive_hourly_parameters = archive_hourly_parameters if archive_hourly_parameters is not None else get_hourly_parameters("ecmwf_h")
        self.forecast_hourly_parameters = forecast_hourly_parameters if forecast_hourly_parameters is not None else get_hourly_parameters("ecmwf_f")

    def get_historical(self,
                       coordinate: Coordinate,
                       start_date: str,
                       end_date: str,
                       columns: Optional[List[str]] = None) -> Response:
        """Make a GET request to the Open Meteo API for historical/archived data.

        Args:
            coordinate (Coordinate): The location to fetch data for.
            start_date (str): The starting date for the requested data. In the format "YYYY-MM-DD".
            end_date (str): The ending date for the requested data. In the format "YYYY-MM-DD".
            columns (list[str], optional): The subset of columns to fetch. If set to None, all columns will be fetched. Defaults to None.

        Returns:
            Response: The response object from the REST API containing response body, headers, status code
        """
        # Setup the Open-Meteo API client with cache and retry on error
        cache_session = requests_cache.CachedSession('.cache', expire_after = -1)
        retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
        openmeteo = openmeteo_requests.Client(session = retry_session)

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

        return openmeteo.weather_api(url, params=params)

    def get_current(self,
                    coordinate: Coordinate,
                    past_days: int = 92,
                    forecast_days: int = 16,
                    columns: Optional[List[str]] = None) -> Response:
        """Make a GET request to the Open Meteo API for current/forecasted data.

        Args:
            coordinate (Coordinate): The location to fetch data for.
            past_days (int, optional): How many days into the past to fetch data for. Defaults to 92 (OpenMeteo max value).
            forecast_days (int, optional): How many days into the future to fetch data for. Defaults to 16 (OpenMeteo max value).
            columns (list[str], optional): The subset of columns to fetch. If set to None, all columns will be fetched. Defaults to None.

        Returns:
            Response: The response object from the REST API containing response body, headers, status code
        """
         # Setup the Open-Meteo API client with cache and retry on error
        cache_session = requests_cache.CachedSession('.cache', expire_after = -1)
        retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
        openmeteo = openmeteo_requests.Client(session = retry_session)

        hourly_params = columns if columns is not None else self.forecast_hourly_parameters

        url = "https://api.open-meteo.com/v1/ecmwf"
        params = {
            "latitude": coordinate.lat,
            "longitude": coordinate.lon,
            "hourly": hourly_params,
            "past_days": 92,
            "forecast_days": 7
        }
        return openmeteo.weather_api(url, params=params)

    def get_index_parameter(self) -> str:
        """Temporal index parameter for OpenMeteo hourly data is "time".

        Returns:
            str: "time"
        """
        return "time"