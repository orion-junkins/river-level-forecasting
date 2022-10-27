from pandas import DataFrame

from rlf.forecasting.data_fetching_utilities.weather_provider.api.base_api_adapter import BaseAPIAdapter
from rlf.forecasting.data_fetching_utilities.coordinate import Coordinate
from rlf.forecasting.data_fetching_utilities.weather_provider.weather_datum import WeatherDatum


class WeatherProvider():
    """Provides a historical of forecasted weather for a given location and time period"""

    def __init__(self, coordinates: Coordinate, api_adapter: BaseAPIAdapter) -> None:
        """Takes a list of coordinates

        Args:
            coordinates (list[Coordinate(longitude: float, latitude: float)]): Named tuple WSG84 coordinates: (longitude, latitude)
            api_adapter (BaseAPIAdapter): An adapter for a weather API
        """
        self.coordinates = coordinates
        self.api_adapter = api_adapter

    def fetch_historical_weather_datums(self, start_date: str, end_date: str) -> list[WeatherDatum]:
        """Fetch historical weather for all coordinates

        Args:
            start_date (str, optional):  iso8601 format YYYY-MM-DD (https://en.wikipedia.org/wiki/ISO_8601).
            end_date (str, optional): iso8601 format YYYY-MM-DD.

        Returns:
            list[WeatherDatum]: A list of WeatherDatum objects containing the weather data and metadata about the location or datum
        """
        datums = []
        for coordinate in self.coordinates:
            datum = self.fetch_historical_weather_at_datum(
                longitude=coordinate.lon, latitude=coordinate.lat, start_date=start_date, end_date=end_date)
            datums.append(datum)
        return datums

    def fetch_historical_weather(self, start_date: str, end_date: str) -> list[DataFrame]:
        """Fetch historical weather for all coordinates

        Args:
            start_date (str, optional):  iso8601 format YYYY-MM-DD (https://en.wikipedia.org/wiki/ISO_8601).
            end_date (str, optional): iso8601 format YYYY-MM-DD.

        Returns:
            list[DataFrame]: A list of DataFrames containing the weather data about the location.
        """
        dfs = []
        for coordinate in self.coordinates:
            datum = self.fetch_historical_weather_at_datum(
                longitude=coordinate.lon, latitude=coordinate.lat, start_date=start_date, end_date=end_date)
            dfs.append(datum.hourly_parameters)
        return dfs

    def fetch_historical_weather_at_datum(self, longitude: float, latitude: float, start_date: str, end_date: str) -> WeatherDatum:
        """Fetch historical weather for a single coordinate or datum

        Args:
            longitude (float): WSG84 longitude
            latitude (float): WSG84 latitude
            start_date (str): iso8601 format YYYY-MM-DD
            end_date (str): iso8601 format YYYY-MM-DD

        Returns:
            WeatherDatum: A Datum object containing the weather data and metadata about a coordinate (https://en.wikipedia.org/wiki/Geodetic_datum)
        """

        self.api_adapter.__dict__.update(
            longitude=longitude, latitude=latitude, start_date=start_date, end_date=end_date)

        response = self.api_adapter.get()

        datum = WeatherDatum(
            longitude=response.data.get(
                "longitude", None),
            latitude=response.data.get(
                "latitude", None),
            elevation=response.data.get(
                "elevation", None),
            utc_offset_seconds=response.data.get(
                "utc_offset_seconds", None),
            timezone=response.data.get(
                "timezone", None),
            hourly_units=response.data.get(
                "hourly_units", None),
            hourly_parameters=response.data.get(
                "hourly", None))

        return datum
