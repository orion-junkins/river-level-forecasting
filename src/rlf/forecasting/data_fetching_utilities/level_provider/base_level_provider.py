from abc import ABC, abstractmethod
from datetime import datetime
from typing import Union

import pandas as pd
import pytz


class BaseLevelProvider(ABC):
    """Abstract base class for Level Providers, agnostic to the underlying source where the data comes from."""

    @abstractmethod
    def fetch_recent_level(self, num_recent_samples: int) -> pd.DataFrame:
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

    def format_level_data(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        """
        Take in a dataframe of level data and handle basic formatting.
            - Convert index to UTC
            - Drop all data excess of hourly frequency
            - Remove duplicates in index
            - Set frequency as hourly
            - Impute NaNs based on adjacent values where possible
            - Drop remaining NaNs

        Args:
            df_raw (pd.DataFrame): Unformatted DataFrame.

        Returns:
            (pd.DataFrame): Formatted DataFrame.
        """
        df_formatted = df_raw.copy()

        # Convert index to utc timestamps
        df_formatted.index = df_formatted.index.map(self._validate_index)

        df_formatted = self._coerce_index_to_hourly(df_formatted)

        # Remove duplicated index entries
        df_formatted = df_formatted[~df_formatted.index.duplicated()]

        # Set frequency as hourly
        # df_formatted = df_formatted.asfreq('H')

        # Compute forward/back filled data
        for_fill = df_formatted.asfreq('H', method='ffill')
        back_fill = df_formatted.asfreq('H', method='bfill')

        # Average the forward and back filled values
        df_formatted = (for_fill + back_fill) / 2

        # Drop any rows remaining which have NaN values (generally first and/or last rows)
        df_formatted.dropna(inplace=True)

        return df_formatted

    @staticmethod
    def _coerce_index_to_hourly(df: pd.DataFrame) -> pd.DataFrame:
        """Coerce the index to hourly by taking the first observed level for each hourly span observed.

        Args:
            df (pd.DataFrame): Pandas DataFrame with a datetime index.

        Returns:
            pd.DataFrame: Pandas DataFrame with hourly observations (original timestamp is overwritten).
        """
        df = (
            df.sort_index()
            .groupby(by=lambda i: i.replace(minute=0, second=0, microsecond=0))
            .first()
        )
        return df

    @staticmethod
    def _validate_index(x: Union[str, datetime]) -> datetime:
        """Validate that an index value is of type datetime and in the utc timezone, and if not then convert it.

        Args:
            x (str | datetime): Index value to validate.

        Returns:
            datetime: Valid index value.
        """
        if isinstance(x, str):
            x = datetime.fromisoformat(x)
        return x.astimezone(pytz.utc)
