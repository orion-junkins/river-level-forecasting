from datetime import datetime
import time
from typing import Optional

from pandas import DataFrame
import pytz


from rlf.forecasting.data_fetching_utilities.coordinate import Coordinate
from rlf.forecasting.data_fetching_utilities.weather_provider.api.base_api_adapter import (
    BaseAPIAdapter
)
from rlf.forecasting.data_fetching_utilities.weather_provider.api.models import Response
from rlf.forecasting.data_fetching_utilities.weather_provider.base_weather_provider import (
    BaseWeatherProvider
)
from rlf.forecasting.data_fetching_utilities.weather_provider.open_meteo.open_meteo_adapter import (
    OpenMeteoAdapter
)
from rlf.forecasting.data_fetching_utilities.weather_provider.weather_datum import (
    WeatherDatum
)


DEFAULT_START_DATE = "2022-01-01"
DEFAULT_END_DATE = datetime.now().strftime("%Y-%m-%d")


class APIWeatherProvider(BaseWeatherProvider):
    """Provides a historical of forecasted weather for a given location and time period."""

    def __init__(self,
                 coordinates: Coordinate,
                 api_adapter: BaseAPIAdapter = OpenMeteoAdapter()) -> None:
        """Create an APIWeatherProvider for the given list of coordinates.

        Args:
            coordinates (list[Coordinate(longitude: float, latitude: float)]): Named tuple WSG84 coordinates: (longitude, latitude).
            api_adapter (BaseAPIAdapter, optional): An adapter for a weather API. Defaults to OpenMeteoAdapter().
        """
        self.coordinates = coordinates
        self.api_adapter = api_adapter

    def _build_hourly_parameters_from_response(self, hourly_parameters_response: dict, tz: str) -> DataFrame:
        index_parameter = self.api_adapter.get_index_parameter()
        df = DataFrame(hourly_parameters_response)
        df.index = df[index_parameter].map(lambda x: datetime.fromisoformat(x).replace(tzinfo=pytz.timezone(tz)).astimezone(pytz.timezone("UTC")))
        df.drop(columns=[index_parameter], inplace=True)
        return df

    def build_datum_from_response(self, response: Response, precision: float = 0.25) -> WeatherDatum:
        """Construct a WeatherDatum from a Response.

        Args:
            response (Response): The Response to draw data from.
            precision (float): The precision for rounding lat and lon. Defaults to 0.25.

        Returns:
            WeatherDatum: The constructed WeatherDatum instance.
        """
        lon = response.data.get("longitude", None)
        lat = response.data.get("latitude", None)
        rounded_lon = round(lon/precision) * precision
        rounded_lat = round(lat/precision) * precision
        datum = WeatherDatum(
            longitude=rounded_lon,
            latitude=rounded_lat,
            elevation=response.data.get(
                "elevation", None),
            utc_offset_seconds=response.data.get(
                "utc_offset_seconds", None),
            timezone=response.data.get(
                "timezone", None),
            hourly_units=response.data.get(
                "hourly_units", None),
            hourly_parameters=self._build_hourly_parameters_from_response(
                response.data.get("hourly", None), response.data["timezone"]))

        return datum

    def fetch_historical_datum(self,
                               coordinate: Coordinate,
                               start_date: str = DEFAULT_START_DATE,
                               end_date: str = DEFAULT_END_DATE,
                               columns: Optional[list[str]] = None) -> WeatherDatum:
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

    def fetch_historical(self,
                         start_date: str = DEFAULT_START_DATE,
                         end_date: str = DEFAULT_END_DATE,
                         columns: Optional[list[str]] = None,
                         sleep_duration: str = 0) -> list[WeatherDatum]:
        """Fetch historical weather for all coordinates.

        Args:
            start_date (str, optional): iso8601 format YYYY-MM-DD. Defaults to DEFAULT_START_DATE.
            end_date (str, optional): iso8601 format YYYY-MM-DD. Defaults to DEFAULT_END_DATE.
            columns (list[str], optional): The columns/parameters to fetch. All available will be fetched if left equal to None. Defaults to None.
            sleep_duration (int, optional): How many seconds to sleep after each query. Helps prevent throttling. Defaults to 0.

        Returns:
            list[WeatherDatum]: A list of WeatherDatum objects containing the weather data and metadata about the locations.
        """
        datums = {}
        for coordinate in self.coordinates:
            datum = self.fetch_historical_datum(
                coordinate=coordinate, start_date=start_date, end_date=end_date, columns=columns)
            coord = Coordinate(datum.longitude, datum.latitude)
            if coord not in datums:
                datums[coord] = datum
            time.sleep(sleep_duration)
        return list(datums.values())

    def fetch_current_datum(self, coordinate: Coordinate, columns: Optional[list[str]] = None) -> WeatherDatum:
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

    def fetch_current(self, columns: Optional[list[str]] = None, sleep_duration: int = 0) -> list[WeatherDatum]:
        """Fetch current weather for all coordinates.

        Args:
            columns (list[str], optional): The columns/parameters to fetch. All available will be fetched if left equal to None. Defaults to None.
            sleep_duration (int, optional): How many seconds to sleep after each query. Helps prevent throttling. Defaults to 0.

        Returns:
            list[WeatherDatum]: A list of WeatherDatums containing the weather data about the location.
        """
        datums = []
        for coordinate in self.coordinates:
            datum = self.fetch_current_datum(
                coordinate=coordinate, columns=columns)
            datums.append(datum)
            time.sleep(sleep_duration)
        return datums
