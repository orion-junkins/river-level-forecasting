from pandas import DataFrame
from datetime import datetime

from rlf.forecasting.data_fetching_utilities.weather_provider.api.base_api_adapter import BaseAPIAdapter
from rlf.forecasting.data_fetching_utilities.weather_provider.api.models import Response
from rlf.forecasting.data_fetching_utilities.weather_provider.open_meteo.open_meteo_adapter import OpenMeteoAdapter
from rlf.forecasting.data_fetching_utilities.coordinate import Coordinate
from rlf.forecasting.data_fetching_utilities.weather_provider.weather_datum import WeatherDatum
from rlf.aws_dispatcher import AWSDispatcher

DEFAULT_START_DATE = "2020-01-01"
DEFAULT_END_DATE = datetime.now().strftime("%Y-%m-%d")


class WeatherProvider():
    """Provides a historical of forecasted weather for a given location and time period."""

    def __init__(self, coordinates: Coordinate, api_adapter: BaseAPIAdapter = OpenMeteoAdapter(), aws_dispatcher: AWSDispatcher = None) -> None:
        """Create a WeatherProvider for the given list of coordinates. Optionally, an AWS dispatcher can be provided, allowing data to be stored/fetched from an S3 bucket rather than re-issuing queries.

        Args:
            coordinates (list[Coordinate(longitude: float, latitude: float)]): Named tuple WSG84 coordinates: (longitude, latitude).
            api_adapter (BaseAPIAdapter): An adapter for a weather API. Defaults to OpenMeteoAdapter().
            aws_dispatcher (AWSDispatcher, optional): An AWSDispatcher for S3 backing if desired. Defaults to None.
        """
        self.coordinates = coordinates
        self.api_adapter = api_adapter
        self.aws_dispatcher = aws_dispatcher

    def _build_hourly_parameters_from_response(self, hourly_parameters_response: dict) -> DataFrame:
        index_parameter = self.api_adapter.get_index_parameter()
        df = DataFrame(hourly_parameters_response)
        df.index = df[index_parameter].map(datetime.fromisoformat)
        df.drop(columns=[index_parameter], inplace=True)
        return df

    def build_datum_from_response(self, response: Response):
        """Construct a WeatherDatum from a Response.

        Args:
            response (Response): The Response to draw data from.

        Returns:
            WeatherDatum: The constructed WeatherDatum instance.
        """
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
            hourly_parameters=self._build_hourly_parameters_from_response(
                response.data.get("hourly", None)))

        return datum

    def fetch_historical_datum(self, coordinate: Coordinate, start_date: str = DEFAULT_START_DATE, end_date: str = DEFAULT_END_DATE, columns: list[str] = None) -> WeatherDatum:
        """
        Fetch historical weather for a single coordinate or datum.

        Args:
            coordinate (Coordinate): The location to fetch data for.
            start_date (str, optional): iso8601 format YYYY-MM-DD. Defaults to DEFAULT_START_DATE.
            end_date (str, optional): iso8601 format YYYY-MM-DD. Defaults to DEFAULT_END_DATE.
            columns (list[str], optional): The columns/parameters to fetch. All available will be fetched if left equal to None. Defaults to None.

        Returns:
            WeatherDatum: A Datum object containing the weather data and metadata about a coordinate.
        """
        response = self.api_adapter.get_historical(coordinate=coordinate, start_date=start_date, end_date=end_date, columns=columns)

        datum = self.build_datum_from_response(response)

        return datum

    def fetch_historical_datums(self, start_date: str = DEFAULT_START_DATE, end_date: str = DEFAULT_END_DATE, columns: list[str] = None) -> list[WeatherDatum]:
        """Fetch historical weather for all coordinates.

        Args:
            start_date (str, optional): iso8601 format YYYY-MM-DD. Defaults to DEFAULT_START_DATE.
            end_date (str, optional): iso8601 format YYYY-MM-DD. Defaults to DEFAULT_END_DATE.
            columns (list[str], optional): The columns/parameters to fetch. All available will be fetched if left equal to None. Defaults to None.

        Returns:
            list[WeatherDatum]: A list of WeatherDatum objects containing the weather data and metadata about the locations.
        """
        datums = []
        for coordinate in self.coordinates:
            datum = self.fetch_historical_datum(
                coordinate=coordinate, start_date=start_date, end_date=end_date, columns=columns)
            datums.append(datum)
        return datums

    def update_historical_datums_in_aws(self, start_date: str = DEFAULT_START_DATE, end_date: str = DEFAULT_END_DATE, columns: list[str] = None) -> None:
        """Refetch historical datums and store this updated data in AWS. This will overwrite whatever data was previously stored for the current river.

        Args:
            start_date (str, optional): iso8601 format YYYY-MM-DD. Defaults to DEFAULT_START_DATE.
            end_date (str, optional): iso8601 format YYYY-MM-DD. Defaults to DEFAULT_END_DATE.
            columns (list[str], optional): The columns/parameters to fetch. All available will be fetched if left equal to None. Defaults to None.

        Raises:
            ValueError: Ensures that an AWSDispatcher has been provided, raising an error if not.
        """
        if self.aws_dispatcher is None:
            raise ValueError("No AWSDispatcher provided.")
        datums = self.fetch_historical_datums(start_date=start_date, end_date=end_date, columns=columns)
        for datum in datums:
            self.aws_dispatcher.upload_datum(datum)

    def download_historical_datums_from_aws(self, columns: list[str] = None) -> None:
        """Download historical datums from AWS. Assumes datums exist in expected location.

        Args:
            columns (list[str], optional): The columns/parameters to fetch. All available will be fetched if left equal to None. Defaults to None.

        Returns:
            list[WeatherDatum]: A list of WeatherDatum objects containing the weather data and metadata about the locations.
        """
        datums = []
        for coordinate in self.coordinates:
            datum = self.aws_dispatcher.download_datum(coordinate, columns=columns)
            datums.append(datum)
        return datums

    def fetch_historical(self, columns: list[str] = None) -> list[DataFrame]:
        """Fetch historical weather for all coordinates. If there is an AWS dispatcher, data will be fetched from there if possible. If there is not a dispatcher, or the AWS file cannot be found, a regular datum request will be issued.

        Args:
            columns (list[str], optional): The columns/parameters to fetch. All available will be fetched if left equal to None. Defaults to None.

        Returns:
            list[DataFrame]: A list of DataFrames containing the weather data about the location.
        """
        if self.aws_dispatcher is None:
            datums = self.fetch_historical_datums(columns=columns)
        else:
            try:
                datums = self.download_historical_datums_from_aws(columns=columns)
            except FileNotFoundError:
                datums = self.fetch_historical_datums(columns=columns)
        dfs = []
        for datum in datums:
            dfs.append(datum.hourly_parameters)
        return dfs

    def fetch_current_datum(self, coordinate: Coordinate, columns: list[str] = None) -> WeatherDatum:
        """Fetch current weather for a single coordinate.

        Args:
            coordinate (Coordinate): The location to fetch data for.
            columns (list[str], optional): The columns/parameters to fetch. All available will be fetched if left equal to None. Defaults to None.

        Returns:
            WeatherDatum: A Datum object containing the weather data and metadata about a coordinate
        """
        response = self.api_adapter.get_current(coordinate=coordinate, columns=columns)

        datum = self.build_datum_from_response(response)

        return datum

    def fetch_current_datums(self, columns: list[str] = None) -> list[WeatherDatum]:
        """Fetch current weather for all coordinates.

        Args:
            columns (list[str], optional): The columns/parameters to fetch. All available will be fetched if left equal to None. Defaults to None.

        Returns:
            list[WeatherDatum]: A list of WeatherDatum objects containing the weather data and metadata about the location or datum
        """
        datums = []
        for coordinate in self.coordinates:
            datum = self.fetch_current_datum(
                coordinate=coordinate, columns=columns)
            datums.append(datum)
        return datums

    def fetch_current(self, columns: list[str] = None) -> list[DataFrame]:
        """Fetch current weather for all coordinates.

        Args:
            columns (list[str], optional): The columns/parameters to fetch. All available will be fetched if left equal to None. Defaults to None.

        Returns:
            list[DataFrame]: A list of DataFrames containing the weather data about the location.
        """
        datums = self.fetch_current_datums(columns=columns)
        dfs = []
        for datum in datums:
            dfs.append(datum.hourly_parameters)
        return dfs
