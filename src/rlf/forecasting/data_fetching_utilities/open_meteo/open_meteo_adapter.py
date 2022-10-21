from rlf.forecasting.data_fetching_utilities.open_meteo.open_meteo_hourly_parameters import OpenMeteoHourlyParameters


class OpenMeteoAdapter():
    """Adapts the OpenMeteo API to be used by the RequestBuilder 
    """

    def __init__(self,
                 protocol: str = "https",
                 hostname: str = "archive-api.open-meteo.com",
                 version: str = "v1",
                 path: str = "era5",
                 latitude: float = 44.06,
                 longitude: float = -121.31,
                 start_date: str = None,
                 end_date: str = None,
                 hourly_parameters: OpenMeteoHourlyParameters().get_variables() = None):
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


        Raises:
            ValueError: If latitude is not between -90 and 90
            ValueError: If longitude is not between -180 and 180
        """
        if latitude < -90 or latitude > 90:
            raise ValueError("latitude must be between -90 and 90")
        if longitude < -180 or longitude > 180:
            raise ValueError("longitude must be between -180 and 180")

        self.protocol = protocol
        self.hostname = hostname
        self.version = version
        self.path = path
        self.latitude = latitude
        self.longitude = longitude
        self.start_date = start_date
        self.end_date = end_date
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
            "latitude": self.latitude,
            "longitude": self.longitude,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "hourly": self.hourly_parameters
        }
        return parameters

    def set_hourly_parameters(self, hourly_parameters: dict) -> None:
        """The hourly parameters to use in a request

        Args:
            hourly_parameters (dict): The parameters to use in a request
        """
        self.hourly_parameters = hourly_parameters

    def set_location(self, latitude: float, longitude: float) -> None:
        """Set the geographical location

        Args:
            latitude (float): Geographical WGS84 latitude
            longitude (float): Geographical WGS84 longitude
        """
        self.latitude = latitude
        self.longitude = longitude

    def get_location(self) -> tuple:
        """Get the geographical location

        Returns:
            tuple: Geographical WGS84 coordinate
        """
        return (self.latitude, self.longitude)

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
