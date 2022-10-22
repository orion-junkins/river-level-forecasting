from abc import ABC, abstractmethod

import pandas as pd


class LevelProvider_ABC(ABC):
    """Abstract base class for Level Providers, agnostic to the underlying source where the data comes from."""

    @abstractmethod
    def fetch_recent_level(self, num_recent_samples) -> pd.DataFrame:
        """Fetch river level data for the most recent num_hours. Dataframe is returned with a tz aware UTC Datetime index.

        Args:
            num_hours (int): Number of hours to fetch data for.

        Returns:
            pd.DataFrame: A dataframe of recent level data with a tz aware UTC Datetime index. Guaranteed to have num_hours rows.
        """
        pass

    @abstractmethod
    def fetch_historical_level(self) -> pd.DataFrame:
        """Fetch all available historical river level data.

        Returns:
            pd.Dataframe: A dataframe of historical level data with a tz aware UTC Datetime index. Guaranteed to have num_hours rows.
        """
        pass
