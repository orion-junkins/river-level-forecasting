from dataclasses import dataclass
import pandas as pd


@dataclass
class WeatherDatum:
    """A datum represents a geographical location via a coordinate system. This class packages data from any datum
        into a single structure containing the information about this point. Each datum can have an interval
        of measurements in hourly time steps. (https://en.wikipedia.org/wiki/Geodetic_datum)

    Args:
        longitude (float): WGS84 datum longitude value between [-180, 180]
        latitude (float): WGS84 datum latitude value between [-90, 90]
        elevation (int): elevation of the location using WGS84
        utc_offset_seconds (float): offset seconds from UTC
        timezone (str): database timezone string
        hourly_units (dict): units of the hourly parameters
        hourly_parameters (dict): contains a key and value for every passed in parameter
    """
    longitude: float
    latitude: float
    elevation: float
    utc_offset_seconds: float
    timezone: str
    hourly_units: dict
    hourly_parameters: dict

    def get_data_frame(self) -> pd.DataFrame:
        """Get the hourly parameters of the location point

        Returns:
            pd.DataFrame: hourly parameters
        """
        return pd.DataFrame(self.hourly_parameters)
