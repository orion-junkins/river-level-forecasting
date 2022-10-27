from datetime import datetime, timedelta

import dataretrieval.nwis as nwis
import pandas as pd

from rlf.forecasting.data_fetching_utilities.level_provider.base_level_provider import BaseLevelProvider


class LevelProviderNWIS(BaseLevelProvider):
    """Provider class for river level data from the USGS NWIS (National Water Information System)."""

    def __init__(self, gauge_id: str) -> None:
        """Create a new level provider for a specific NWIS gauge.

        Args:
            gauge_id (str): A string of the USGS gauge id number.
        """
        self.gauge_id = gauge_id

    def fetch_recent_level(self, num_hours: int) -> pd.DataFrame:
        """Fetch river level data for the most recent num_hours. Dataframe is returned with a tz aware UTC Datetime index.

        Args:
            num_hours (int): Number of hours to fetch data for.

        Returns:
            pd.DataFrame: A dataframe of recent level data with a tz aware UTC Datetime index. Guaranteed to have num_hours rows.
        """
        start_dt = datetime.utcnow() - timedelta(hours=(num_hours+48))
        start_str = datetime.strftime(start_dt, '%Y-%m-%d')
        data_all = self.fetch_level(start=start_str)
        data_trimmed = data_all.iloc[-num_hours:, :]

        return data_trimmed

    def fetch_historical_level(self) -> pd.DataFrame:
        """Fetch all historical level data from the beginning of collection to the most recent available data. Dataframe is returned with a tz aware UTC Datetime index.

        Returns:
            pd.Dataframe: A dataframe of historical level data with a tz aware UTC Datetime index. Guaranteed to have num_hours rows.
        """
        return self.fetch_level()

    def fetch_level(self, start: str = "1900-01-01", end: str = None, parameterCd: str = '00060', drop_cols: list[str] = ["00060_cd", "site_no"], rename_dict: dict = {"00060": "level"}) -> pd.DataFrame:
        """
        Fetch level data for the given gauge ID. Fetches instant values from start to end.
        Drops and renames columns according to given args.
        Args:
            start (str, optional): Start date in the form "yyyy-mm-dd". Defaults to "1900-01-01", giving data from start of collection.
            end  (str, optional): End date in the form "yyyy-mm-dd". Defaults to None, giving data til end of collection.
            parameterCd (str, optional): Which parameter to fetch data for. Defaults to '00060' indicated mean level.
            drop_cols (list, optional): Column names to drop if they are present. Defaults to ["00060_cd", "site_no"] (useless metadata).
            rename_dict (dict, optional): Dictionary of default:new defining column renamings. Defaults to {"00060":"level"}.
        Returns:
            df (Pandas dataframe): Formatted dataframe of fetched data
        """
        # Fetch level data
        df = nwis.get_record(sites=self.gauge_id, service='iv', start=start, end=end, parameterCd=parameterCd)

        # Filter out any columns that are present in the drop_cols list
        drop_cols = list(filter(lambda x: x in df.columns, drop_cols))
        df.drop(columns=drop_cols, inplace=True)

        # Rename columns as specified
        df.rename(columns=rename_dict, inplace=True)

        # Format data
        df = BaseLevelProvider.format_level_data(df)

        return df
