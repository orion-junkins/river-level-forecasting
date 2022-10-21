from rlf.forecasting.data_fetching_utilities.weather_provider.api.api import RequestBuilder
from rlf.forecasting.data_fetching_utilities.weather_provider.open_meteo.models import ResponseModel
from rlf.forecasting.data_fetching_utilities.weather_provider.open_meteo.open_meteo_adapter import OpenMeteoAdapter
from rlf.forecasting.data_fetching_utilities.weather_provider.datum import Datum
from rlf.forecasting.data_fetching_utilities.coordinate import Coordinate


class WeatherProvider():
    """Provides a historical of forecasted weather for a given location and time period"""

    def __init__(self, coordinates: Coordinate):
        """Takes a list of coordinates

        Args:
            coordinates (list[Coordinate: (float, float)]): Named tuple WSG84 coordinates: (longitude, latitude)
        """
        self.coordinates = coordinates

    def fetch_historical_weather(self, start_date: str, end_date: str) -> list[Datum]:
        """Fetch historical weather for all coordinates

        Args:
            start_date (str, optional):  iso8601 format YYYY-MM-DD (https://en.wikipedia.org/wiki/ISO_8601). Defaults to "2021-01-01".
            end_date (str, optional): iso8601 format YYYY-MM-DD. Defaults to "2021-01-02".

        Returns:
            list[Datum]: A list of Datum objects containing the weather data and metadata about the location or datum
        """
        datums = []
        for coordinate in self.coordinates:
            datum = self.fetch_historical_weather_at_datum(
                longitude=coordinate.lon, latitude=coordinate.lat, start_date=start_date, end_date=end_date)
            datums.append(datum)
        return datums

    def fetch_historical_weather_at_datum(self, longitude: float, latitude: float, start_date: str, end_date: str) -> Datum:
        """Fetch historical weather for a single coordinate or datum

        Args:
            longitude (float): WSG84 longitude
            latitude (float): WSG84 latitude
            start_date (str): iso8601 format YYYY-MM-DD
            end_date (str): iso8601 format YYYY-MM-DD

        Returns:
            Datum: A Datum object containing the weather data and metadata about a coordinate (https://en.wikipedia.org/wiki/Geodetic_datum)
        """        """"""
        open_meteo_archive_api_adapter = OpenMeteoAdapter(hostname="archive-api.open-meteo.com", latitude=latitude, longitude=longitude,
                                                          start_date=start_date, end_date=end_date,)

        request_builder = RequestBuilder(
            api_adapter=open_meteo_archive_api_adapter)

        response = request_builder.get()

        response_model = ResponseModel(**response.data)

        datum = Datum(longitude=response_model.longitude, latitude=response_model.latitude,
                      elevation=response_model.elevation, utc_offset_seconds=response_model.utc_offset_seconds,
                      timezone=response_model.timezone, hourly_parameters=response_model.hourly_parameters())

        return datum
