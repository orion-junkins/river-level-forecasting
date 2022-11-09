from abc import ABC, abstractmethod

from rlf.forecasting.data_fetching_utilities.coordinate import Coordinate
from rlf.forecasting.data_fetching_utilities.weather_provider.api.models import Response


class BaseAPIAdapter(ABC):
    """Abstract base class for APIAdapter objects"""

    @abstractmethod
    def get_current(self, coordinate: Coordinate, **kwargs) -> Response:
        """Get the current/forecast payload for the request in the form of a hash map

        Args:
            coordinate (Coordinate): The location to fetch data for.

        Returns:
            Response: The response payload for the request
        """
        return NotImplementedError

    @abstractmethod
    def get_historical(self, coordinate: Coordinate, **kwargs) -> Response:
        """Get the historical/archive payload for the request in the form of a hash map

        Args:
            coordinate (Coordinate): The location to fetch data for.

        Returns:
            Response: The response payload for the request
        """
        return NotImplementedError
