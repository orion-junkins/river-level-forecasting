from dataclasses import dataclass

import pandas as pd


@dataclass
class WeatherDatum:
    """A datum represents data associated with a single geographical location via a coordinate system. This class packages data from any datum into a single structure containing the information about this point. Each datum can have an interval of measurements in hourly time steps.

    Args:
        longitude (float): the requested WGS84 datum longitude value between [-180, 180]
        latitude (float): the requested WGS84 datum latitude value between [-90, 90]
        api_response_longitude (float): the longitude value returned by the API
        api_response_latitude (float): the latitude value returned by the API
        elevation (int): elevation of the location using WGS84
        utc_offset_seconds (float): offset seconds from UTC
        timezone (str): database timezone string
        hourly_units (dict): units of the hourly parameters
        hourly_parameters (pd.DataFrame): DataFrame of all hourly parameter data
    """
    longitude: float
    latitude: float
    api_response_longitude: float
    api_response_latitude: float
    elevation: float
    utc_offset_seconds: float
    timezone: str
    hourly_units: dict
    hourly_parameters: pd.DataFrame

    @property
    def meta_data(self) -> dict:
        """Meta data associated with the Datum contained in a single dictionary.

        Returns:
            dict: All metadata (attributes excluding hourly_units and hourly_parameters).
        """
        return {"longitude": self.longitude,
                "latitude": self.latitude,
                "api_response_longitude": self.api_response_longitude,
                "api_response_latitude": self.api_response_latitude,
                "elevation": self.elevation,
                "utc_offset_seconds": self.utc_offset_seconds,
                "timezone": self.timezone}
