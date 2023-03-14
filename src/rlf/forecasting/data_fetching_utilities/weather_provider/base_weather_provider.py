from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Optional


from rlf.forecasting.data_fetching_utilities.coordinate import Coordinate
from rlf.forecasting.data_fetching_utilities.weather_provider.weather_datum import (
    WeatherDatum
)

DEFAULT_START_DATE = "2022-01-01"
DEFAULT_END_DATE = datetime.now().strftime("%Y-%m-%d")

current_parameter_remaps_from_adapter = {
    "soil_temperature_0_to_10cm": "soil_temperature_level_1",
    "soil_temperature_10_to_40cm": "soil_temperature_level_2",
    "soil_temperature_40_to_100cm": "soil_temperature_level_3",
    "soil_temperature_100_to_200cm": "soil_temperature_level_4",
    "soil_moisture_0_to_10cm": "soil_moisture_level_1",
    "soil_moisture_10_to_40cm": "soil_moisture_level_2",
    "soil_moisture_40_to_100cm": "soil_moisture_level_3",
    "soil_moisture_100_to_200cm": "soil_moisture_level_4",
}

current_parameter_remaps_to_adapter = {value: key for key, value in current_parameter_remaps_from_adapter.items()}

historical_parameter_remaps_from_adapter = {
    "soil_temperature_0_to_7cm": "soil_temperature_level_1",
    "soil_temperature_7_to_28cm": "soil_temperature_level_2",
    "soil_temperature_28_to_100cm": "soil_temperature_level_3",
    "soil_temperature_100_to_255cm": "soil_temperature_level_4",
    "soil_moisture_0_to_7cm": "soil_moisture_level_1",
    "soil_moisture_7_to_28cm": "soil_moisture_level_2",
    "soil_moisture_28_to_100cm": "soil_moisture_level_3",
    "soil_moisture_100_to_255cm": "soil_moisture_level_4"
}

historical_parameter_remaps_to_adapter = {value: key for key, value in historical_parameter_remaps_from_adapter.items()}


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

    def _remap_current_parameters_to_adapter(self, params: List[str]) -> List[str]:
        """Remap the parameter names for current data from the consistent names to the adapter's actual names.

        Args:
            params (List[str]): Initial list of params to remap.

        Returns:
            List[str]: New list with param names either remapped or left alone (maintains order).
        """
        return [current_parameter_remaps_to_adapter.get(param, param) for param in params]

    def _remap_current_parameters_from_adapter(self, params: List[str]) -> List[str]:
        """Remap the parameter names for current data from the adapter's actual name to the consistent names.

        Args:
            params (List[str]): Initial list of params to remap.

        Returns:
            List[str]: New list with param names either remapped or left alone (maintains order).
        """
        return [current_parameter_remaps_from_adapter.get(param, param) for param in params]

    def _remap_historical_parameters_to_adapter(self, params: List[str]) -> List[str]:
        """Remap the parameter names for historical data from the consistent names to the adapter's actual names.

        Args:
            params (List[str]): Initial list of params to remap.

        Returns:
            List[str]: New list with param names either remapped or left alone (maintains order).
        """
        return [historical_parameter_remaps_to_adapter.get(param, param) for param in params]

    def _remap_historical_parameters_from_adapter(self, params: List[str]) -> List[str]:
        """Remap the parameter names for historical data from the adapter's actual names to the consistent names.

        Args:
            params (List[str]): Initial list of params to remap.

        Returns:
            List[str]: New list with param names either remapped or left alone (maintains order).
        """
        return [historical_parameter_remaps_from_adapter.get(param, param) for param in params]
