from rlf.forecasting.data_fetching_utilities.weather_provider.api.api_adapter_abc import APIAdapter_ABC
from rlf.forecasting.data_fetching_utilities.weather_provider.open_meteo.parameters import get_hourly_parameters


class OpenMeteoAdapter(APIAdapter_ABC):
    """Adapts the OpenMeteo API to be used by the RequestBuilder"""

    def __init__(self,
                 longitude: float,
                 latitude: float,
                 start_date: str,
                 end_date: str,
                 protocol: str = "https",
                 hostname: str = "archive-api.open-meteo.com",
                 version: str = "v1",
                 path: str = "era5",
                 hourly_parameters: list[str] = get_hourly_parameters()) -> None:
        """Adapts the OpenMeteo API to be used by the RequestBuilder

        Args:
            protocol (str, optional): The protocol to use. Defaults to "https".
            hostname (str, optional): The hostname to use. Defaults to "archive-api.open-meteo.com".
            version (str, optional): The version of the API to use. Defaults to "v1".
            path (str, optional): The path to use. Defaults to "era5".
            latitude (float, optional): Geographical WGS84 coordinate. Defaults to 44.06. Bend, OR
            longitude (float, optional): Geographical WGS84 coordinate. Defaults to -121.31. Bend, OR
            start_date (str, optional): ISO8601 date (yyyy-mm-dd). Defaults to None.
            end_date (str, optional): ISO8601 date (yyyy-mm-dd). Defaults to None.
            hourly_parameters (OpenMeteoHourlyParameters, optional): The parameters to use. Defaults to None.
        """
        self.longitude = longitude
        self.latitude = latitude
        self.start_date = start_date
        self.end_date = end_date
        self.protocol = protocol
        self.hostname = hostname
        self.version = version
        self.path = path
        self.hourly_parameters = hourly_parameters

    def get_payload(self) -> dict:
        """Returns: The payload to use in the request

        Returns:
            dict: The payload to use in a request
        """
        payload = {
            "protocol": self.protocol,
            "hostname": self.hostname,
            "version": self.version,
            "path": self.path,
            "parameters": self.get_parameters()
        }
        return payload

    def get_parameters(self) -> dict:
        """The parameters to use in a request

        Returns:
            dict: The parameters to use in a request
        """
        parameters = {
            "longitude": self.longitude,
            "latitude": self.latitude,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "hourly": self.hourly_parameters
        }
        return parameters

    def set_location(self, longitude: float, latitude: float) -> None:
        """Set the geographical location

        Args:
            longitude (float): Geographical WGS84 longitude
            latitude (float): Geographical WGS84 latitude
        """
        self.longitude = longitude
        self.latitude = latitude

    def get_location(self) -> tuple:
        """Get the geographical location

        Returns:
            tuple: Geographical WGS84 coordinate
        """
        return (self.longitude, self.latitude)

    def set_start_date(self, start_date: str) -> None:
        """Start date for the request

        Args:
            start_date (str): IS08601 date (yyyy-mm-dd)
        """
        self.start_date = start_date

    def get_start_date(self) -> str:
        """Get the start date for the request

        Returns:
            str: IS08601 date (yyyy-mm-dd)
        """
        return self.start_date

    def set_end_date(self, end_date: str) -> None:
        """Set the end date for the request

        Args:
            end_date (str): IS08601 date (yyyy-mm-dd)
        """
        self.end_date = end_date

    def get_end_date(self) -> str:
        """Get the end date for the request

        Returns:
            str: IS08601 date (yyyy-mm-dd)
        """
        return self.end_date
