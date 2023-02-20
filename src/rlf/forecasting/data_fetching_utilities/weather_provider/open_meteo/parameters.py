from typing import List

HOURLY_PARAMETERS = {"era5": [
    "temperature_2m",
    "relativehumidity_2m",
    "dewpoint_2m",
    "apparent_temperature",
    "pressure_msl",
    "surface_pressure",
    "precipitation",
    "rain",
    "snowfall",
    "cloudcover",
    "cloudcover_low",
    "cloudcover_mid",
    "cloudcover_high",
    "shortwave_radiation",
    "direct_radiation",
    "diffuse_radiation",
    "direct_normal_irradiance",
    "windspeed_10m",
    "windspeed_100m",
    "winddirection_10m",
    "winddirection_100m",
    "windgusts_10m",
    "et0_fao_evapotranspiration",
    "vapor_pressure_deficit",
    "soil_temperature_0_to_7cm",
    "soil_temperature_7_to_28cm",
    "soil_temperature_28_to_100cm",
    "soil_temperature_100_to_255cm",
    "soil_moisture_0_to_7cm",
    "soil_moisture_7_to_28cm",
    "soil_moisture_28_to_100cm",
    "soil_moisture_100_to_255cm"],

    "gfs": [
    "temperature_2m",
    "relativehumidity_2m",
    "dewpoint_2m",
    "apparent_temperature",
    "pressure_msl",
    "surface_pressure",
    "precipitation",
    "snowfall",
    "snow_depth",
    "freezinglevel_height",
    "visibility",
    "cloudcover",
    "cloudcover_low",
    "cloudcover_mid",
    "cloudcover_high",
    "evapotranspiration",
    "et0_fao_evapotranspiration",
    "vapor_pressure_deficit",
    "cape",
    "lifted_index",
    "windspeed_10m",
    "windspeed_80m",
    "winddirection_10m",
    "winddirection_80m",
    "windgusts_10m",
    "soil_temperature_0_to_10cm",
    "soil_temperature_10_to_40cm",
    "soil_temperature_40_to_100cm",
    "soil_temperature_100_to_200cm",
    "soil_moisture_0_to_10cm",
    "soil_moisture_10_to_40cm",
    "soil_moisture_40_to_100cm",
    "soil_moisture_100_to_200cm"
    ]}


def get_hourly_parameters(param_list="era5") -> List[str]:
    """The parameters to use in a request used by the Open Meteo API

    Returns:
        list: The hourly parameters to use in a request which are weather measurements
    """
    return HOURLY_PARAMETERS[param_list]
