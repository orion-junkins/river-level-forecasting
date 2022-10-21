from forecasting.data_fetching_utilities.api.api import RequestBuilder
from forecasting.data_fetching_utilities.open_meteo.models import ResponseModel
from forecasting.data_fetching_utilities.open_meteo.open_meteo_adapter import OpenMeteoAdapter
from forecasting.data_fetching_utilities.open_meteo.open_meteo_hourly_parameters import OpenMeteoHourlyParameters
from forecasting.data_fetching_utilities.data_providers.datum import Datum


class WeatherProvider():
    """Provides a historical of forecasted weather for a given location and time period"""

    def __init__(self, locations: list[tuple: float, float]):
        """Takes a list of tuples of longitude and latitude

        Args:
            locations (list[tuple: (float, float)]): WSG84 coordinates: (longitude, latitude)
        """
        self.locations = locations

    def fetch_historical_weather(self, start_date: str = "2021-01-01", end_date: str = "2021-01-02") -> list[Datum]:
        """Fetch historical weather for all locations

        Args:
            start_date (str, optional):  iso8601 format YYYY-MM-DD (https://en.wikipedia.org/wiki/ISO_8601). Defaults to "2021-01-01".
            end_date (str, optional): iso8601 format YYYY-MM-DD. Defaults to "2021-01-02".

        Returns:
            list[Datum]: A list of Datum objects containing the weather data and metadata about the location or datum
        """
        datums = []
        for location in self.locations:
            datum = self.fetch_historical_weather_at_datum(
                longitude=location[0], latitude=location[1], start_date=start_date, end_date=end_date)
            datums.append(datum)
        return datums

    def fetch_historical_weather_at_datum(self, longitude: float, latitude: float, start_date: str, end_date: str) -> Datum:
        """Fetch historical weather for a single location or datum

        Args:
            longitude (float): WSG84 longitude
            latitude (float): WSG84 latitude
            start_date (str): iso8601 format YYYY-MM-DD
            end_date (str): iso8601 format YYYY-MM-DD

        Returns:
            Datum: A Datum object containing the weather data and metadata about a location (https://en.wikipedia.org/wiki/Geodetic_datum)
        """        """"""
        open_meteo_archive_api_adapter = OpenMeteoAdapter(hostname="archive-api.open-meteo.com", latitude=latitude, longitude=longitude,
                                                          start_date=start_date, end_date=end_date,)

        hourly_parameters = OpenMeteoHourlyParameters()
        hourly_parameters.set_all()

        open_meteo_archive_api_adapter.set_hourly_parameters(
            hourly_parameters=hourly_parameters.get_variables())

        request_builder = RequestBuilder(
            api_adapter=open_meteo_archive_api_adapter)

        response = request_builder.get()
        print("Message:", response.message,
              "Status Code:", response.status_code)

        response_model = ResponseModel(**response.data)

        datum = Datum(longitude=response_model.longitude, latitude=response_model.latitude,
                      elevation=response_model.elevation, utc_offset_seconds=response_model.utc_offset_seconds,
                      timezone=response_model.timezone, hourly_parameters=response_model.hourly_parameters())

        return datum
