from datetime import datetime
from typing import List, Optional

import pandas as pd
import pytz

from rlf.aws_dispatcher import AWSDispatcher
from rlf.forecasting.data_fetching_utilities.weather_provider.base_weather_provider import (
    BaseWeatherProvider
)

DEFAULT_START_DATE = "2022-01-01"
DEFAULT_END_DATE = datetime.now().strftime("%Y-%m-%d")


class AWSWeatherUploader():
    """Utility for fetching and storing data from some WeatherProvider into AWS. Generally used to make data accessible to AWSWeatherProvider instances"""

    def __init__(self,
                 weather_provider: BaseWeatherProvider,
                 aws_dispatcher: AWSDispatcher) -> None:
        """
        Create a new AWSWeatherUploader instance.

        Args:
            weather_provider (BaseWeatherProvider): Source for fetching data.
            aws_dispatcher (AWSDispatcher): An AWSDispatcher to upload data to.
        """
        self.weather_provider = weather_provider
        self.aws_dispatcher = aws_dispatcher

    def upload_historical(self,
                          start_date: str = DEFAULT_START_DATE,
                          end_date: str = DEFAULT_END_DATE,
                          columns: Optional[List[str]] = None,
                          years_per_query: int = 10,
                          sleep_duration: int = 0) -> None:
        """Refetch historical datums and store this updated data in AWS. This will overwrite whatever data was previously stored for the current river.

        Args:
            start_date (str, optional): iso8601 format YYYY-MM-DD. Expected in UTC. Defaults to DEFAULT_START_DATE.
            end_date (str, optional): iso8601 format YYYY-MM-DD. Expected in UTC. Defaults to DEFAULT_END_DATE.
            columns (list[str], optional): The columns/parameters to fetch. All available will be fetched if left equal to None. Defaults to None.
            years_per_query (int, optional): How many years to fetch in a single query. Defaults to 2.
            sleep_duration (int, optional): How long to sleep after each query. Helps prevent throttling. Defaults to 0.
        """
        start_datetime = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=pytz.UTC)
        end_datetime = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=pytz.UTC)

        partition_end_date = min(
            [start_datetime.replace(start_datetime.year + years_per_query), end_datetime])

        datums = self.weather_provider.fetch_historical(
            start_date=datetime.strftime(start_datetime, "%Y-%m-%d"),
            end_date=datetime.strftime(partition_end_date, "%Y-%m-%d"),
            columns=columns,
            sleep_duration=sleep_duration)

        start_datetime = start_datetime.replace(start_datetime.year + years_per_query)
        partition_end_date = partition_end_date.replace(
            partition_end_date.year + years_per_query)

        while partition_end_date < end_datetime:
            partial_datums = self.weather_provider.fetch_historical(
                start_date=datetime.strftime(start_datetime, "%Y-%m-%d"),
                end_date=datetime.strftime(partition_end_date, "%Y-%m-%d"),
                columns=columns,
                sleep_duration=sleep_duration)

            for (datum, partial_datum) in zip(datums, partial_datums):
                partial_hourly_parameters = partial_datum.hourly_parameters
                datum.hourly_parameters = pd.concat([datum.hourly_parameters,
                                                    partial_hourly_parameters])

            start_datetime = start_datetime.replace(start_datetime.year + years_per_query)
            partition_end_date = partition_end_date.replace(
                partition_end_date.year + years_per_query)

        for datum in datums:
            self.aws_dispatcher.upload_datum(datum, "historical")

    def upload_current(self,
                       columns: Optional[List[str]] = None,
                       sleep_duration: float = 0.0,
                       dir_path: Optional[str] = None) -> None:
        """Refetch current datums and store this updated data in AWS. This will overwrite whatever data was previously stored for the current river.

        Args:
            columns (list[str], optional): The columns/parameters to fetch. All available will be fetched if left equal to None. Defaults to None.
            sleep_duration (float, optional): How long to sleep after each query. Helps prevent throttling. Defaults to 0.0.
            dir_path (str, optional): The subdir (within 'current') to store datums. Generally set equal to the timestamp of collection. Defaults to None.
        """
        if dir_path is None:
            dir_path = "current"
        else:
            dir_path = f'current/{dir_path}'

        datums = self.weather_provider.fetch_current(
            columns=columns,
            sleep_duration=sleep_duration)

        for datum in datums:
            self.aws_dispatcher.upload_datum(datum, dir_path=dir_path)
