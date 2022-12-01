from datetime import datetime
from typing import Optional
import pytz


from rlf.aws_dispatcher import AWSDispatcher
from rlf.forecasting.data_fetching_utilities.coordinate import Coordinate

from rlf.forecasting.data_fetching_utilities.weather_provider.base_weather_provider import (
    BaseWeatherProvider
)
from rlf.forecasting.data_fetching_utilities.weather_provider.weather_datum import (
    WeatherDatum
)


DEFAULT_START_DATE = "2022-01-01"
DEFAULT_END_DATE = datetime.now().strftime("%Y-%m-%d")


class AWSWeatherProvider(BaseWeatherProvider):
    """Provides a historical of forecasted weather for a given location and time period. Backed by AWS. This makes queries to a provided AWS dispatcher for all data."""

    def __init__(self,
                 coordinates: Coordinate,
                 aws_dispatcher: AWSDispatcher) -> None:
        """Create an APIWeatherProvider for the given list of coordinates.

        Args:
            coordinates (list[Coordinate(longitude: float, latitude: float)]): Named tuple WSG84 coordinates: (longitude, latitude).
            aws_dispatcher (AWSDispatcher): The AWSDispatcher instance from which data will be drawn.
        """
        self.coordinates = coordinates
        self.aws_dispatcher = aws_dispatcher

    def download_historical_datums_from_aws(self, columns: Optional[list[str]] = None) -> list[WeatherDatum]:
        """Download historical datums from AWS. Assumes datums exist in expected location.

        Args:
            columns (list[str], optional): The columns/parameters to fetch. All available will be fetched if left equal to None. Defaults to None.

        Raises:
            FileNotFoundError: If no datum can be found at the provided weather

        Returns:
            list[WeatherDatum]: A list of WeatherDatum objects containing the weather data and metadata about the locations.
        """
        datums = []
        for coordinate in self.coordinates:
            datum = self.aws_dispatcher.download_datum(
                coordinate, columns=columns)
            datums.append(datum)
        return datums

    def fetch_historical(self, columns: Optional[list[str]] = None, start_date: str = None, end_date: str = None) -> list[WeatherDatum]:
        """Fetch historical weather for all coordinates. If there is an AWS dispatcher, data will be fetched from there if possible. If there is not a dispatcher, or the AWS file cannot be found, a regular datum request will be issued.

        Args:
            columns (list[str], optional): The columns/parameters to fetch. All available will be fetched if left equal to None. Defaults to None.

        Returns:
            list[WeatherDatum]: A list of WeatherDatums containing the weather data about the location.
        """
        datums = self.download_historical_datums_from_aws(columns=columns)

        if start_date is not None:
            start_dt = datetime.datetime.strptime(
                start_date, '%Y-%m-%d').replace(tzinfo=pytz.UTC)

            if end_date is not None:
                end_dt = datetime.datetime.strptime(
                    start_date, '%Y-%m-%d').replace(tzinfo=pytz.UTC)
                for datum in datums:
                    datum.hourly_parameters = datum.hourly_parameters[start_dt:end_dt]
            else:
                datum.hourly_parameters = datum.hourly_parameters[start_dt:]

        return datums

    def fetch_current(self, columns: Optional[list[str]] = None) -> list[WeatherDatum]:
        """Fetch current weather for all coordinates.

        Args:
            columns (list[str], optional): The columns/parameters to fetch. All available will be fetched if left equal to None. Defaults to None.

        Returns:
            list[WeatherDatum]: A list of WeatherDatums containing the weather data about the location.
        """
        raise (
            NotImplementedError)  # TODO Implement fetching of evaluation snapshots as "current"
