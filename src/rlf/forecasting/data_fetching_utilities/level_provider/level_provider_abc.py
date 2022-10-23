from abc import ABC, abstractmethod
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

    @staticmethod
    def format_level_data(df_raw: pd.DataFrame) -> pd.DataFrame:
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
        df_formatted.index = df_formatted.index.map(lambda x: x.astimezone(pytz.utc))

        if df_formatted.index[0].minute != 0:
            df_formatted.drop([df_formatted.index[0]], inplace=True)
        if not (df_formatted.index[0].minute == 0):
            raise Exception("Error: failed to coerce index to hourly.")

        # Remove duplicated entries
        df_formatted = df_formatted.loc[~df_formatted.index.duplicated(), :]

        # Set frequency as hourly
        df_formatted = df_formatted.asfreq('H')

        # Compute forward/back filled data
        for_fill = df_formatted.fillna(method='ffill')
        back_fill = df_formatted.fillna(method='bfill')
        # For every column in the dataframe,
        for col in df_formatted.columns:
            # Average the forward and back filled values
            df_formatted[col] = (for_fill[col] + back_fill[col])/2

        # Drop any rows remaining which have NaN values (generally first and/or last rows)
        df_formatted.dropna(inplace=True)

        return df_formatted
