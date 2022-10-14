from forecasting.data_fetching_utilities.api.api import RequestBuilder
from forecasting.data_fetching_utilities.open_meteo.models import ResponseModel
from forecasting.data_fetching_utilities.open_meteo.open_meteo_adapter import OpenMeteoAdapter
from forecasting.data_fetching_utilities.open_meteo.hourly import Hourly

import pandas as pd

TEST_DATE_MIN = "2021-01-01"
TEST_DATE_MAX = "2021-01-02"


class WeatherProvider():

    def __init__(self, locations: list[tuple: float, float]):
        self.locations = locations

    def fetch_historical_weather(self) -> pd.DataFrame:
        open_meteo_archive = OpenMeteoAdapter(hostname="archive-api.open-meteo.com", latitude=self.locations[0][0], longitude=self.locations[0][1],
                                              start_date=TEST_DATE_MIN, end_date=TEST_DATE_MAX,)
        hourly = Hourly()
        hourly.set_all()

        open_meteo_archive.set_hourly(hourly=hourly.get_variables())
        request_builder = RequestBuilder(api_adapter=open_meteo_archive)
        response = request_builder.get()
        print("Message:", response.message,
              "Status Code:", response.status_code)

        response_model = ResponseModel(**response.data)
        df = pd.DataFrame(response_model.hourly.__dict__)
        df["latitude"] = response_model.latitude
        df["longitude"] = response_model.longitude
        df["generationtime_ms"] = response_model.generationtime_ms
        df["utc_offset_seconds"] = response_model.utc_offset_seconds
        df["timezone"] = response_model.timezone
        df["timezone_abbreviation"] = response_model.timezone_abbreviation
        df["elevation"] = response_model.elevation
        return df

    def fetch_current_weather(self, past_hours_to_fetch, future_hours_to_fetch) -> pd.DataFrame:
        pass
