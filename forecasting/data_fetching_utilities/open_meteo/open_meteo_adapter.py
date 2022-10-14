from forecasting.data_fetching_utilities.open_meteo.hourly import Hourly


class OpenMeteoAdapter():

    def __init__(self,
                 protocol: str = "https",
                 hostname: str = "archive-api.open-meteo.com",
                 version: str = "v1",
                 path: str = "era5",
                 latitude: float = 44.06,
                 longitude: float = -121.31,
                 start_date: str = None,
                 end_date: str = None,
                 hourly: Hourly().get_variables() = None):
        """Default Location: Bend, OR (44.06, -121.31)"""

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
        self.hourly = hourly

    def get_payload(self) -> dict:
        payload = {
            "protocol": self.protocol,
            "hostname": self.hostname,
            "version": self.version,
            "path": self.path,
            "parameters": self.get_parameters()
        }
        return payload

    def get_parameters(self) -> dict:
        parameters = {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "hourly": self.hourly
        }
        return parameters

    def set_hourly(self, hourly: dict) -> None:
        self.hourly = hourly

    def set_location(self, latitude: float, longitude: float) -> None:
        """Args: Geographical WGS84 coordinate"""
        self.latitude = latitude
        self.longitude = longitude

    def get_location(self) -> float:
        """Returns: Geographical WGS84 coordinate"""
        return (self.latitude, self.longitude)

    def set_start_date(self, start_date: str) -> None:
        """Args: ISO8601 date (yyyy-mm-dd)"""
        self.start_date = start_date

    def get_start_date(self) -> str:
        """Returns: ISO8601 date (yyyy-mm-dd)"""
        return self.start_date

    def set_end_date(self, end_date: str) -> None:
        """Args: IS08601 date (yyyy-mm-dd)"""
        self.end_date = end_date

    def get_end_date(self) -> str:
        """Returns: IS08601 date (yyyy-mm-dd)"""
        return self.end_date
