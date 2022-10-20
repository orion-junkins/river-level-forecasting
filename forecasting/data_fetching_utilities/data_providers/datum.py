import pandas as pd


class Datum():

    def __init__(self, longitude: float, latitude: float, elevation: int,
                 utc_offset_seconds: float, timezone: str, hourly_parameters: dict):
        """A datum represents a geographical location via a coordinate system. This class packages data from any datum 
            into a single structure containing the information about this point. Each datum can have an interval 
            of measurements in hourly time steps.

        Args:
            longitude (float): value between [-180, 180]
            latitude (float): value between [-90, 90]
            elevation (int): elevation of the location using WGS84
            utc_offset_seconds (float): offset seconds from UTC
            timezone (str): database timezone string
            hourly_parameters (dict): contains a key and value for every passed in parameter 
        """
        self.longitude = longitude
        self.latitude = latitude
        self.elevation = elevation
        self.utc_offset_seconds = utc_offset_seconds
        self.timezone = timezone
        self.hourly_parameters = hourly_parameters

    def get_hourly_parameters(self, as_pandas_data_frame: bool = True) -> pd.DataFrame or dict:
        """Get the hourly parameters of the location point

        Args:
            as_pandas_data_frame (bool, optional): True to build a DataFrame. Defaults to True.

        Returns:
            pd.DataFrame or dict: choose from dataframe or dictionary
        """        """"""
        if as_pandas_data_frame:
            return pd.DataFrame(self.hourly_parameters)
        else:
            return self.hourly_parameters

    def get_longitude(self) -> float:
        """WGS84 datum longitude

        Returns:
            float: longitude
        """
        return self.longitude

    def get_latitude(self) -> float:
        """WGS84 datum latitude

        Returns:
            float: latitude
        """
        return self.latitude

    def get_elevation(self) -> int:
        """Elevation in passed in units
        Returns:
            int: elevation
        """
        return self.elevation

    def get_utc_offset_seconds(self) -> float:
        """UTC offset seconds

        Returns:
            float: utc offset seconds
        """
        return self.utc_offset_seconds

    def get_timezone(self) -> str:
        """Timezone string

        Returns:
            str: timezone
        """
        return self.timezone
