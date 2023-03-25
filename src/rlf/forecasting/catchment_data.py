from typing import List, Optional, Tuple
from pandas import DataFrame

from rlf.forecasting.data_fetching_utilities.level_provider.base_level_provider import BaseLevelProvider
from rlf.forecasting.data_fetching_utilities.weather_provider.base_weather_provider import BaseWeatherProvider


class CatchmentData:
    """Abstraction for containing all data pertaining to a single catchment. Acts as a cache for fetched data with the ability to update/refetch when desired. Data is not fetched until first access. Thus, this class may be used in production environments for inference without loading excess historical data."""

    def __init__(
        self,
        catchment_name: str,
        weather_provider: BaseWeatherProvider,
        level_provider: BaseLevelProvider,
        num_recent_samples: int = 90*24,
        columns: Optional[List[str]] = None
    ) -> None:
        """
        Create a CatchmentData instance.

        Args:
            catchment_name (str): Name of the catchment. Should correspond to external gauge name.
            weather_provider (BaseWeatherProvider): Provider for Weather data.
            level_provider (BaseLevelProvider): Provider for level data.
            num_recent_samples (int, optional): Number of recent level samples to fetch. Defaults to 90 days (90 days * 24 hours/day).
            columns (list[str], optional): The columns/parameters to fetch. All available will be fetched if left equal to None. Defaults to None.
        """
        self.name = catchment_name
        self.weather_provider = weather_provider
        self.level_provider = level_provider
        self.num_recent_samples = num_recent_samples
        self.columns = columns

        self._all_current: Optional[Tuple[List[DataFrame], DataFrame]] = None  # (weather_data, level_data)
        self._all_historical = (None, None)  # (weather_data, level_data)

    @property
    def num_weather_datasets(self) -> int:
        """The number of datasets in the currently fetched weather data. Guaranteed to be the same for current and historical data. Corresponds to the number of weather locations.

        Raises:
            ValueError: If there is a mismatch in the number of datasets between current and historical weather data.

        Returns:
            int: The number of weather datasets.
        """
        current_weather = self.all_current[0]
        historical_weather = self.all_historical[0]
        if (len(current_weather) != len(historical_weather)):
            raise ValueError("Must have the same number of historical and current datasets")
        return len(current_weather)

    @property
    def all_current(self) -> Tuple[List[DataFrame], DataFrame]:
        """All current data. A tuple containing a list of current weather data with one DataFrame per location, and a single DataFrame of level data.

        Returns:
            tuple[list[DataFrame], DataFrame]: All current data for the Catchment.
        """
        if self._all_current is None:
            self._fetch_all_current()
        assert self._all_current is not None
        return self._all_current

    def _fetch_all_current(self) -> None:
        """Fetch or refetch all current data, updating the member variable _all_current. This will trigger queries to the underlying weather and level providers."""
        current_weather = self.weather_provider.fetch_current(columns=self.columns)
        recent_level = self.level_provider.fetch_recent_level(self.num_recent_samples)

        self._all_current = (current_weather, recent_level)

    def update_for_inference(self) -> None:
        """External facing helping to update current data for inference."""
        self._fetch_all_current()

    @property
    def all_historical(self) -> Tuple[List[DataFrame], DataFrame]:
        """All historical data. A tuple containing a list of historical weather data with one DataFrame per location, and a single DataFrame of level data.

        Returns:
            tuple[list[DataFrame], DataFrame]: All historical data for the Catchment.
        """
        if self._all_historical == (None, None):
            self._fetch_all_historical()

        return self._all_historical

    def _fetch_all_historical(self) -> None:
        """Fetch or refetch all current data, updating the member variable _all_current. This will trigger queries to the underlying weather and level providers."""
        historical_level = self.level_provider.fetch_historical_level()
        earliest_historical_level = historical_level.index.to_series().min().strftime("%Y-%m-%d")
        historical_weather = self.weather_provider.fetch_historical(start_date=earliest_historical_level, columns=self.columns)
        self._all_historical = (historical_weather, historical_level)
