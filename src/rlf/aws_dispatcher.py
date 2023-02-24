import json
from typing import List
import os

from pandas import DataFrame
import pyarrow.parquet as pq
import s3fs

from rlf.forecasting.data_fetching_utilities.coordinate import Coordinate
from rlf.forecasting.data_fetching_utilities.weather_provider.weather_datum import WeatherDatum
from typing import Optional


DEFAULT_LOCAL_PATH = os.path.join("data", "aws_dispatch")
os.makedirs(DEFAULT_LOCAL_PATH, exist_ok=True)


class AWSDispatcher():
    def __init__(self, bucket_name: str, directory_name: str) -> None:
        """Create a new AWS Dispatcher instance.

        Args:
            bucket_name (str): The target bucket for dispatching. MUST already exist in AWS.
            directory_name (str): Directory name within the target bucket. Does not need to already exist.
        """
        self.s3 = s3fs.S3FileSystem(anon=False)
        self.working_dir = f's3://{bucket_name}/{directory_name}'

    def upload_as_json(self, dictionary: dict, folder_name: str, filename: str) -> None:
        """
        Pickle the given dictionary locally, and upload the file to AWS

        Args:
            dictionary (dict): dictionary to write to S3.
            folder_name (str): desired folder for the file. Determines s3 folder name.
            filename (str): desired name for the file. Determines s3 file name.
        """
        path = f'{self.working_dir}/{folder_name}/{filename}.json'

        self.s3.write_text(
            value=json.dumps(dictionary),
            path=path
        )

    def download_dict_from_json(self, folder_name: str, filename: str) -> dict:
        """Download a json file from AWS and parse it into a dictionary.

        Args:
            folder_name (str): Folder of the file.
            filename (str): Name for the file.

        Raises:
            FileNotFoundError: Raised if the file cannot be found at the expected path in AWS.

        Returns:
            dict: The resulting dictionary.
        """
        path = f'{self.working_dir}/{folder_name}/{filename}.json'
        try:
            with self.s3.open(path, 'rb') as f:
                data = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError("Could not find a json file at path: " + path)

        return data

    def upload_as_parquet(self, dataframe: DataFrame, folder_name: str, filename: str) -> None:
        """
        Pickle the given dictionary locally, and upload the file to AWS

        Args:
            dataframe (DataFrame): DataFrame to upload to S3.
            folder_name (str): desired folder for the file. Determines s3 folder name.
            filename (str): desired name for the file. Determines s3 file name.
        """
        path = f'{self.working_dir}/{folder_name}/{filename}.parquet'

        self.s3.write_bytes(
            value=dataframe.to_parquet(),
            path=path
        )

    def download_df_from_parquet(self, folder_name: str, filename: str, columns: Optional[List[str]] = None) -> DataFrame:
        """Download a parquet file from AWS and parse it into a DataFRame

        Args:
            folder_name (str): Folder of the file.
            filename (str): Name for the file.
            columns (list[str], optional): Columns to fetch. All available will be fetched if set to None. Defaults to None.

        Raises:
            FileNotFoundError: Raised if the file cannot be found at the expected path in AWS.

        Returns:
            DataFrame: Downloaded DataFrame.
        """
        path = f'{self.working_dir}/{folder_name}/{filename}.parquet'
        try:
            dataset = pq.ParquetDataset(path, filesystem=self.s3)
            if columns is not None and 'time' not in columns:
                columns.append('time')
            table = dataset.read(columns=columns)
            df = table.to_pandas()
        except FileNotFoundError:
            raise FileNotFoundError("Could not find a parquet file at path: " + path)
        return df

    def upload_datum(self, datum: WeatherDatum, dir_path: Optional[str] = None) -> None:
        """Upload an entire WeatherDatum to S3.

        Args:
            datum (WeatherDatum): The Datum to upload.
            dir_path (str): Directory path to which datum should be uploaded.
        """
        if dir_path is None:
            folder_name = f'lon_{datum.longitude:.2f}_lat_{datum.latitude:.2f}'
        else:
            folder_name = f'{dir_path}/lon_{datum.longitude:.2f}_lat_{datum.latitude:.2f}'

        self.upload_as_json(datum.meta_data, folder_name, "meta")
        self.upload_as_json(datum.hourly_units, folder_name, "units")
        self.upload_as_parquet(datum.hourly_parameters, folder_name, "data")

    def download_datum(self, coordinate: Coordinate, columns: Optional[List[str]] = None,  dir_path: Optional[str] = None) -> WeatherDatum:
        """Download an entire WeatherDatum from S3.

        Args:
            coordinate (Coordinate): Coordinate to fetch Datum for.
            columns (list[str], optional): Columns to fetch. All available will be fetched if set to None. Defaults to None.
            dir_path (str): Directory path to which datum should be uploaded.

        Raises:
            FileNotFoundError: Raised if any needed files cannot be found at the expected paths in AWS.

        Returns:
            WeatherDatum: Downloaded WeatherDatum
        """
        folder_name = f'{dir_path}/lon_{coordinate.lon:.2f}_lat_{coordinate.lat:.2f}'
        try:
            meta_data = self.download_dict_from_json(folder_name, "meta")
            hourly_units = self.download_dict_from_json(folder_name, "units")
            if columns is not None:
                hourly_units = dict((key, hourly_units[key]) for key in columns)
            hourly_parameters = self.download_df_from_parquet(folder_name, "data", columns=columns)
        except FileNotFoundError as e:
            print("Error occured while fetching saved datum from AWS for coordinate: " + str(coordinate))
            raise e
        datum = WeatherDatum(hourly_units=hourly_units, hourly_parameters=hourly_parameters, **meta_data)
        return datum
