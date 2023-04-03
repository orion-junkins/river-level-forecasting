from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Optional


from rlf.forecasting.data_fetching_utilities.coordinate import Coordinate
from rlf.forecasting.data_fetching_utilities.weather_provider.weather_datum import (
    WeatherDatum
)

DEFAULT_START_DATE = "2022-01-01"
DEFAULT_END_DATE = datetime.now().strftime("%Y-%m-%d")


class BaseWeatherProvider(ABC):
    """Provides historical and forecasted weather for a given set of locations. WeatherProviders exist at a single moment in time. Relative to that moment, they provide access to current (recent + forecasted) weather data as well as historical (beginning of collection to some point in the past) weather data."""

    def __init__(self, coordinates: List[Coordinate]) -> None:
        """Create a WeatherProvider for the given list of coordinates.

        Args:
            coordinates (list[Coordinate(longitude: float, latitude: float)]): Named tuple WSG84 coordinates: (longitude, latitude).
        """
        self.coordinates = coordinates

    @abstractmethod
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
            list[WeatherDatum]: A list of WeatherDatums containing the weather data about the location.
        """
        pass

    @abstractmethod
    def fetch_current(self,
                      columns: Optional[List[str]] = None,
                      sleep_duration: float = 0.0) -> List[WeatherDatum]:
        """Fetch current weather for all coordinates.

        Args:
            columns (list[str], optional): The columns/parameters to fetch. All available will be fetched if left equal to None. Defaults to None.
            sleep_duration (float, optional): How many seconds to sleep after each query. Helps prevent throttling. Defaults to 0.0.

        Returns:
            list[WeatherDatum]: A list of WeatherDatums containing the weather data about the location.
        """
        pass
