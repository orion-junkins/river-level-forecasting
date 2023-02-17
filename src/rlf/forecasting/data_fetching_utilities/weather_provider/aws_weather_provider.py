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
                 aws_dispatcher: AWSDispatcher,
                 current_timestamp: Optional[str] = None) -> None:
        """Create an APIWeatherProvider for the given list of coordinates.

        Args:
            coordinates (list[Coordinate(longitude: float, latitude: float)]): Named tuple WSG84 coordinates: (longitude, latitude).
            aws_dispatcher (AWSDispatcher): The AWSDispatcher instance from which data will be drawn.
            current_timestamp (str): The 'current' timestamp for which current data will be fetched. Expected in the form "YY-mm-DD_HH-MM" in UTC. Expected to match a directory in the current weather dir for the AWSProvider.
        """
        self.coordinates = coordinates
        self.aws_dispatcher = aws_dispatcher
        self.current_timestamp = current_timestamp

    def download_datums_from_aws(self, dir_path: str, columns: Optional[list[str]] = None) -> list[WeatherDatum]:
        """Download datums from AWS. Assumes datums exist in expected location.

        Args:
            dir_path (str): The directory path relative to the working directory of the aws_dispatcher.
            columns (list[str], optional): The columns/parameters to fetch. All available will be fetched if left equal to None. Defaults to None.

        Raises:
            FileNotFoundError: If no datum can be found at the provided weather

        Returns:
            list[WeatherDatum]: A list of WeatherDatum objects containing the weather data and metadata about the locations.
        """
        datums = []
        for coordinate in self.coordinates:
            datum = self.aws_dispatcher.download_datum(
                coordinate, columns=columns, dir_path=dir_path)
            datums.append(datum)
        return datums

    def fetch_historical(self,
                         columns: Optional[list[str]] = None,
                         start_date: Optional[str] = None,
                         end_date: Optional[str] = None,
                         sleep_duration: float = 0.0) -> list[WeatherDatum]:
        """Fetch historical weather for all coordinates. If there is an AWS dispatcher, data will be fetched from there if possible. If there is not a dispatcher, or the AWS file cannot be found, a regular datum request will be issued.

        Args:
            columns (list[str], optional): The columns/parameters to fetch. All available will be fetched if left equal to None. Defaults to None.
            start_date (str): The starting date for the requested data. In the format "YYYY-MM-DD".
            end_date (str): The ending date for the requested data. In the format "YYYY-MM-DD".
            sleep_duration (float, optional): How many seconds to sleep after each query. Helps prevent throttling. Defaults to 0.0.

        Returns:
            list[WeatherDatum]: A list of WeatherDatums containing the weather data about the location.
        """
        if sleep_duration != 0.0:
            raise ValueError("sleep_duration is not supported (and generally not needed) for aws_weather_provider")

        datums = self.download_datums_from_aws(dir_path="historical", columns=columns)

        if start_date is not None:
            start_dt = datetime.strptime(
                start_date, '%Y-%m-%d').replace(tzinfo=pytz.UTC)
        else:
            start_dt = None
        if end_date is not None:
            end_dt = datetime.strptime(
                end_date, '%Y-%m-%d').replace(tzinfo=pytz.UTC)
        else:
            end_dt = None
        for datum in datums:
            datum.hourly_parameters = datum.hourly_parameters[start_dt:end_dt]

        return datums

    def fetch_current(self, columns: Optional[list[str]] = None, sleep_duration: float = 0.0) -> list[WeatherDatum]:
        """Fetch current weather for all coordinates.

        Args:
            columns (list[str], optional): The columns/parameters to fetch. All available will be fetched if left equal to None. Defaults to None.
            sleep_duration (float, optional): How many seconds to sleep after each query. Helps prevent throttling. Defaults to 0.0.

        Returns:
            list[WeatherDatum]: A list of WeatherDatums containing the weather data about the location.
        """
        if sleep_duration != 0.0:
            raise ValueError("sleep_duration is not supported (and generally not needed) for aws_weather_provider")

        if self.current_timestamp is None:
            raise ValueError("Cannot fetch current data without a timestamp.")

        dir_path = f'current/{self.current_timestamp}'
        datums = self.download_datums_from_aws(dir_path=dir_path, columns=columns)

        return datums

    def set_timestamp(self, new_timestamp: str) -> None:
        """Set the current timestamp for the weather provider. Fetched "current" weather will be relative to this point in time. Expected to be a valid directory in AWS.

        Args:
            new_timestamp (str): Timestamp in the format "YY-mm-DD_HH-MM" in UTC. Expected to match a directory in the current weather dir for the AWSProvider.
        """
        self.current_timestamp = new_timestamp
