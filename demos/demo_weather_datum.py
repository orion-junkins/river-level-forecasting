# %%
from rlf.forecasting.data_fetching_utilities.weather_provider.api.api import RequestBuilder
from rlf.forecasting.data_fetching_utilities.weather_provider.open_meteo.open_meteo_adapter import OpenMeteoAdapter
from rlf.forecasting.data_fetching_utilities.weather_provider.weather_datum import WeatherDatum

# %%
latitude = 52.52
longitude = 30
start_date = "2021-01-01"
end_date = "2021-01-02"

# %%
open_meteo_archive_api_adapter = OpenMeteoAdapter(
    hostname="archive-api.open-meteo.com",
    latitude=latitude,
    longitude=longitude,
    start_date=start_date,
    end_date=end_date)

request_builder = RequestBuilder(
    api_adapter=open_meteo_archive_api_adapter)
response = request_builder.get()

# %%
datum = WeatherDatum(longitude=response.data["longitude"],
                     latitude=response.data["latitude"],
                     elevation=response.data["elevation"],
                     utc_offset_seconds=response.data["utc_offset_seconds"],
                     timezone=response.data["timezone"],
                     hourly_units=response.data["hourly_units"],
                     hourly_parameters=response.data["hourly"])
print(datum.get_data_frame())
# %%
