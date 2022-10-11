
from datetime import date

from forecasting.data_fetching_utilities.historical_weather.weather_api_hourly_parameter import WeatherAPIHourlyParameter


class WeatherAPIParameters:

    def __init__(self, latitude: float = 44.06, longitude: float = -121.31, start_date: str = date.today().isoformat(),
                 end_date: str = date.today().isoformat(), hourly_parameter: WeatherAPIHourlyParameter() = None):
        """Default Location: Bend, OR

        Args: latitude (float): (decimal degrees)
                longitude (float): (decimal degrees)
                start_date (str): (yyyy-mm-dd)
                end_date (str): (yyyy-mm-dd)
                hourly_parameter (WeatherAPIHourlyParameter): (WeatherAPIHourlyParameter)

        Raises: ValueError: if latitude is not between -90 and 90
                ValueError: if longitude is not between -180 and 180"""
        if latitude < -90 or latitude > 90:
            raise ValueError("latitude must be between -90 and 90")
        if longitude < -180 or longitude > 180:
            raise ValueError("longitude must be between -180 and 180")
        self.latitude = latitude
        self.longitude = longitude
        self.start_date = start_date
        self.end_date = end_date
        self.hourly_parameter = hourly_parameter

    def set_location(self, latitude: float, longitude: float) -> None:
        """Set the Geographical WGS84 coordinate of the location

        Args: latitude (float): (decimal degrees)
                longitude (float): (decimal degrees)

        Returns: None"""
        self.latitude = latitude
        self.longitude = longitude

    def get_location(self) -> float:
        """Get the Geographical WGS84 coordinate of the location

        Returns: (float, float): (latitude, longitude)"""
        return (self.latitude, self.longitude)

    def set_start_date(self, start_date: str) -> None:
        """Set the ISO8601 start date for the weather data to be fetched.

        Args: start_date (str): (yyyy-mm-dd)

        Returns: None"""
        self.start_date = start_date

    def get_start_date(self) -> str:
        """Get the start date for the weather data to be fetched.

        Args: None

        Returns: start_date (str): (yyyy-mm-dd)"""
        return self.start_date

    def set_end_date(self, end_date: str) -> None:
        """Set the ISO8601 end date for the weather data to be fetched.

        Args: end_date (str): (yyyy-mm-dd)

        Returns: None"""
        self.end_date = end_date

    def get_end_date(self) -> str:
        """Get the end date for the weather data to be fetched.

        Args: None

        Returns: end_date (str): (yyyy-mm-dd)"""
        return self.end_date

    def get_hourly_parameter(self) -> list[str]:
        """Get the hourly parameter's weather variables to build the hourly query string.

        Args: None

        Returns: list[str]: list of weather variables"""
        if self.hourly_parameter is not None:
            return self.hourly_parameter.get_weather_variable_names()
        else:
            return None
