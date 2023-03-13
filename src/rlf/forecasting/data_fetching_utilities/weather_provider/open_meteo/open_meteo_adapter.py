from typing import Any, Dict, List, Optional

from rlf.forecasting.data_fetching_utilities.coordinate import Coordinate
from rlf.forecasting.data_fetching_utilities.weather_provider.api.base_api_adapter import BaseAPIAdapter
from rlf.forecasting.data_fetching_utilities.weather_provider.api.models import Response
from rlf.forecasting.data_fetching_utilities.weather_provider.api.rest_invoker import RestInvoker
from rlf.forecasting.data_fetching_utilities.weather_provider.open_meteo.parameters import (
    CURRENT_PARAMETER_MAPS_FROM_API,
    CURRENT_PARAMETER_MAPS_TO_API,
    get_hourly_parameters,
    HISTORICAL_PARAMETER_MAPS_FROM_API,
    HISTORICAL_PARAMETER_MAPS_TO_API,
)


class OpenMeteoAdapter(BaseAPIAdapter):
    """Adapts the OpenMeteo API to be used by the RequestBuilder"""

    def __init__(self,
                 protocol: str = "https",
                 archive_hostname: str = "archive-api.open-meteo.com",
                 forecast_hostname: str = "api.open-meteo.com",
                 version: str = "v1",
                 archive_path: str = "era5",
                 forecast_path: str = "gfs",
                 archive_hourly_parameters: Optional[List[str]] = None,
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
        self.protocol = protocol
        self.archive_hostname = archive_hostname
        self.forecast_hostname = forecast_hostname
        self.version = version
        self.archive_path = archive_path
        self.forecast_path = forecast_path
        self.archive_hourly_parameters = archive_hourly_parameters if archive_hourly_parameters is not None else get_hourly_parameters(archive_path)
        self.forecast_hourly_parameters = forecast_hourly_parameters if forecast_hourly_parameters is not None else get_hourly_parameters(forecast_path)

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
        invoker = RestInvoker(protocol=self.protocol,
                              hostname=self.archive_hostname, version=self.version)

        hourly_params = (columns if columns is not None else self.archive_hourly_parameters).copy()
        hourly_params = [
            HISTORICAL_PARAMETER_MAPS_TO_API[param] if param in HISTORICAL_PARAMETER_MAPS_TO_API else param
            for param in hourly_params
        ]

        parameters = {
            "longitude": coordinate.lon,
            "latitude": coordinate.lat,
            "elevation": "nan",
            "start_date": start_date,
            "end_date": end_date,
            "hourly": hourly_params,
            "cell_selection": "nearest",
        }

        response = invoker.get(path=self.archive_path, parameters=parameters)

        self._remap_response_parameters(response.data, HISTORICAL_PARAMETER_MAPS_FROM_API)

        return response

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
        invoker = RestInvoker(protocol=self.protocol,
                              hostname=self.forecast_hostname, version=self.version)

        hourly_params = (columns if columns is not None else self.forecast_hourly_parameters).copy()
        hourly_params = [
            CURRENT_PARAMETER_MAPS_TO_API[param] if param in CURRENT_PARAMETER_MAPS_TO_API else param
            for param in hourly_params
        ]

        parameters = {
            "longitude": coordinate.lon,
            "latitude": coordinate.lat,
            "elevation": "nan",
            "past_days": past_days,
            "forecast_days": forecast_days,
            "hourly": hourly_params,
            "cell_selection": "nearest",
        }

        response = invoker.get(path=self.forecast_path, parameters=parameters)

        self._remap_response_parameters(response.data, CURRENT_PARAMETER_MAPS_FROM_API)

        return response

    def get_index_parameter(self) -> str:
        """Temporal index parameter for OpenMeteo hourly data is "time".

        Returns:
            str: "time"
        """
        return "time"

    def _remap_response_parameters(self, response_data: Dict[str, Any], parameter_map: Dict[str, str]) -> None:
        """Remap response parameters using the provided parameter map.

        Any parameter found in response_data["hourly"] that is found in parameter_map
        will be renamed based on the mapping found in parameter_map.

        Args:
            response_data (Dict[str, Any]): Response.data dictionary to remap.
            parameter_map (Dict[str, str]): Parameter name mappings to apply to response_data.
        """
        if "hourly" in response_data and response_data["hourly"] is not None:
            response_params = list(response_data["hourly"].keys())
            for response_param in response_params:
                if response_param in parameter_map:
                    remapped_column = parameter_map[response_param]
                    response_data["hourly"][remapped_column] = response_data["hourly"][response_param]
                    del response_data["hourly"][response_param]
