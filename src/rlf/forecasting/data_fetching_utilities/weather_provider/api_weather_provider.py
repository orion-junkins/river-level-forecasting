from datetime import datetime
import logging
import time
from typing import List, Optional
import pandas as pd
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
from rlf.forecasting.data_fetching_utilities.weather_provider.open_meteo.ecmwf_adapter import (
    OpenMeteoECMWFAdapter
)
from rlf.forecasting.data_fetching_utilities.weather_provider.weather_datum import (
    WeatherDatum
)


DEFAULT_START_DATE = "2022-01-01"
DEFAULT_END_DATE = datetime.now().strftime("%Y-%m-%d")

RESPONSE_TOLERANCE = 0.25


class APIWeatherProvider(BaseWeatherProvider):
    """Provides a historical of forecasted weather for a given location and time period."""

    def __init__(self,
                 coordinates: List[Coordinate],
                 api_adapter: BaseAPIAdapter = OpenMeteoECMWFAdapter()) -> None:
        """Create an APIWeatherProvider for the given list of coordinates.

        Args:
            coordinates (list[Coordinate(longitude: float, latitude: float)]): Named tuple WSG84 coordinates: (longitude, latitude).
            api_adapter (BaseAPIAdapter, optional): An adapter for a weather API. Defaults to OpenMeteoAdapter().
        """
        self.coordinates = coordinates
        self.api_adapter = api_adapter



    def _build_hourly_parameters_from_response(self, hourly, ...) -> DataFrame:
        # This is the function that needs to be figure out
        return df
    
    def build_datum_from_response(self, response, coordinate: Coordinate, precision = 5) -> WeatherDatum:
        """Construct a WeatherDatum from a Response.

        Args:
            response (WeatherApiResponse): The response from the API.
            coordinate (Coordinate): The coordinate that is requested by the user.
            precision (int, optional): The number of decimal places to round the response coordinates to. Defaults to 5.

        Returns:
            WeatherDatum: The constructed WeatherDatum instance.
        """
        # Issue some query to get a response from the api
        requested_lon = coordinate.lon
        requested_lat = coordinate.lat

        response_lon = response.Longitude()
        response_lat = response.Latitude()

        response_rounded_lon = round(response_lon, precision)
        response_rounded_lat = round(response_lat, precision)

        difference_rounded_lon = response_rounded_lon - requested_lon
        difference_rounded_lat = response_rounded_lat - requested_lat

        if abs(difference_rounded_lon) > RESPONSE_TOLERANCE or abs(difference_rounded_lat) > RESPONSE_TOLERANCE:
            logging.error(
                "The API responded with a location outside the requested location tolerance. "
                f"The requested location is ({requested_lon}, {requested_lat}) vs. the response location ({response_lon}, {response_lat}). "
                f"The difference in longitude is {difference_rounded_lon} and the difference in latitude is {difference_rounded_lat}. "
                "To change the tolerance, change the RESPONSE_TOLERANCE constant in the APIWeatherProvider class. "
                "To change the rounding precision, change the precision argument in the build_datum_from_response method.")

        datum = WeatherDatum(
            longitude=requested_lon,
            latitude=requested_lat,
            api_response_longitude=response_lon,
            api_response_latitude=response_lat,
            elevation=response.Elevation(),
            utc_offset_seconds=response.Hourly(),
            timezone=response.Timezone(),
            hourly_units=None,
            hourly_parameters=self._build_hourly_parameters_from_response(response.Hourly())
            
        )

        return datum

    def fetch_historical_datum(self,
                               coordinate: Coordinate,
                               start_date: str = DEFAULT_START_DATE,
                               end_date: str = DEFAULT_END_DATE,
                               columns: Optional[List[str]] = None) -> WeatherDatum:
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

        datum = self.build_datum_from_response(response, coordinate)

        return datum

    def fetch_historical(self,
                         columns: Optional[List[str]] = None,
                         start_date: str = DEFAULT_START_DATE,
                         end_date: str = DEFAULT_END_DATE,
                         sleep_duration: float = 0.0) -> List[WeatherDatum]:
        """Fetch historical weather for all coordinates.

        Args:
            columns (list[str], optional): The columns/parameters to fetch. All available will be fetched if left equal to None. Defaults to None.
            start_date (str, optional): iso8601 format YYYY-MM-DD. Defaults to DEFAULT_START_DATE.
            end_date (str, optional): iso8601 format YYYY-MM-DD. Defaults to DEFAULT_END_DATE.
            sleep_duration (float, optional): How many seconds to sleep after each query. Helps prevent throttling. Defaults to 0.0.

        Returns:
            list[WeatherDatum]: A list of WeatherDatum objects containing the weather data and metadata about the locations.
        """
        datums = {}

        for coordinate in self.coordinates:
            datum = self.fetch_historical_datum(coordinate=coordinate, start_date=start_date, end_date=end_date, columns=columns)
            coord = Coordinate(datum.longitude, datum.latitude)
            if coord not in datums:
                datums[coord] = datum
            time.sleep(sleep_duration)
        return list(datums.values())

    def fetch_current_datum(self, coordinate: Coordinate, columns: Optional[List[str]] = None) -> WeatherDatum:
        """Fetch current weather for a single coordinate.

        Args:
            coordinate (Coordinate): The location to fetch data for.
            columns (list[str], optional): The columns/parameters to fetch. All available will be fetched if left equal to None. Defaults to None.

        Returns:
            WeatherDatum: A Datum object containing the weather data and metadata about a coordinate
        """
        response = self.api_adapter.get_current(coordinate=coordinate, columns=columns)

        datum = self.build_datum_from_response(response, coordinate)

        return datum

    def fetch_current(self, columns: Optional[List[str]] = None, sleep_duration: float = 0.0) -> List[WeatherDatum]:
        """Fetch current weather for all coordinates.

        Args:
            columns (list[str], optional): The columns/parameters to fetch. All available will be fetched if left equal to None. Defaults to None.
            sleep_duration (float, optional): How many seconds to sleep after each query. Helps prevent throttling. Defaults to 0.0.

        Returns:
            list[WeatherDatum]: A list of WeatherDatums containing the weather data about the location.
        """
        datums = []

        for coordinate in self.coordinates:
            datum = self.fetch_current_datum(coordinate=coordinate, columns=columns)
            datums.append(datum)
            time.sleep(sleep_duration)
        return datums
