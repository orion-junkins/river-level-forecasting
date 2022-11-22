from rlf.forecasting.data_fetching_utilities.coordinate import Coordinate
from rlf.forecasting.data_fetching_utilities.weather_provider.api.base_api_adapter import BaseAPIAdapter
from rlf.forecasting.data_fetching_utilities.weather_provider.open_meteo.parameters import get_hourly_parameters
from rlf.forecasting.data_fetching_utilities.weather_provider.api.models import Response
from rlf.forecasting.data_fetching_utilities.weather_provider.api.rest_invoker import RestInvoker


class OpenMeteoAdapter(BaseAPIAdapter):
    """Adapts the OpenMeteo API to be used by the RequestBuilder"""

    def __init__(self,
                 protocol: str = "https",
                 archive_hostname: str = "archive-api.open-meteo.com",
                 forecast_hostname: str = "api.open-meteo.com",
                 version: str = "v1",
                 archive_path: str = "era5",
                 forecast_path: str = "gfs",
                 archive_hourly_parameters: list[str] = None,
                 forecast_hourly_parameters: list[str] = None) -> None:
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

    def get_historical(self, coordinate: Coordinate, start_date: str, end_date: str, columns: list[str] = None) -> Response:
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

        hourly_params = columns if columns is not None else self.archive_hourly_parameters

        parameters = {
            "longitude": coordinate.lon,
            "latitude": coordinate.lat,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": hourly_params
        }

        return invoker.get(path=self.archive_path, parameters=parameters)

    def get_current(self, coordinate: Coordinate, past_days: int = 92, forecast_days: int = 16, columns: list[str] = None) -> Response:
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

        hourly_params = columns if columns is not None else self.forecast_hourly_parameters
        parameters = {
            "longitude": coordinate.lon,
            "latitude": coordinate.lat,
            "past_days": past_days,
            "forecast_days": forecast_days,
            "hourly": hourly_params
        }

        return invoker.get(path=self.forecast_path, parameters=parameters)

    def get_index_parameter(self) -> str:
        """Temporal index parameter for OpenMeteo hourly data is "time".

        Returns:
            str: "time"
        """
        return "time"
