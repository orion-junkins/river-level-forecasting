from typing import Optional


class HourlyModel():

    def __init__(self,
                 time: Optional[list[str]] = None,
                 temperature_2m: Optional[list[float]] = None,
                 relativehumidity_2m: Optional[list[int]] = None,
                 dewpoint_2m: Optional[list[float]] = None,
                 apparent_temperature: Optional[list[float]] = None,
                 pressure_msl: Optional[list[float]] = None,
                 surface_pressure: Optional[list[float]] = None,
                 precipitation: Optional[list[float]] = None,
                 rain: Optional[list[float]] = None,
                 snowfall: Optional[list[float]] = None,
                 cloudcover: Optional[list[int]] = None,
                 cloudcover_low: Optional[list[int]] = None,
                 cloudcover_mid: Optional[list[int]] = None,
                 cloudcover_high: Optional[list[int]] = None,
                 shortwave_radiation: Optional[list[int]] = None,
                 direct_radiation: Optional[list[int]] = None,
                 diffuse_radiation: Optional[list[int]] = None,
                 direct_normal_irradiance: Optional[list[float]] = None,
                 windspeed_10m: Optional[list[float]] = None,
                 windspeed_100m: Optional[list[float]] = None,
                 winddirection_10m: Optional[list[int]] = None,
                 winddirection_100m: Optional[list[int]] = None,
                 windgusts_10m: Optional[list[float]] = None,
                 et0_fao_evapotranspiration: Optional[list[float]] = None,
                 vapor_pressure_deficit: Optional[list[float]] = None,
                 soil_temperature_0_to_7cm: Optional[list[float]] = None,
                 soil_temperature_7_to_28cm: Optional[list[float]] = None,
                 soil_temperature_28_to_100cm: Optional[list[float]] = None,
                 soil_temperature_100_to_255cm: Optional[list[float]] = None,
                 soil_moisture_0_to_7cm: Optional[list[float]] = None,
                 soil_moisture_7_to_28cm: Optional[list[float]] = None,
                 soil_moisture_28_to_100cm: Optional[list[float]] = None,
                 soil_moisture_100_to_255cm: Optional[list[float]] = None
                 ):
        self.time = time
        self.temperature_2m = temperature_2m
        self.relativehumidity_2m = relativehumidity_2m
        self.dewpoint_2m = dewpoint_2m
        self.apparent_temperature = apparent_temperature
        self.pressure_msl = pressure_msl
        self.surface_pressure = surface_pressure
        self.precipitation = precipitation
        self.rain = rain
        self.snowfall = snowfall
        self.cloudcover = cloudcover
        self.cloudcover_low = cloudcover_low
        self.cloudcover_mid = cloudcover_mid
        self.cloudcover_high = cloudcover_high
        self.shortwave_radiation = shortwave_radiation
        self.direct_radiation = direct_radiation
        self.diffuse_radiation = diffuse_radiation
        self.direct_normal_irradiance = direct_normal_irradiance
        self.windspeed_10m = windspeed_10m
        self.windspeed_100m = windspeed_100m
        self.winddirection_10m = winddirection_10m
        self.winddirection_100m = winddirection_100m
        self.windgusts_10m = windgusts_10m
        self.et0_fao_evapotranspiration = et0_fao_evapotranspiration
        self.vapor_pressure_deficit = vapor_pressure_deficit
        self.soil_temperature_0_to_7cm = soil_temperature_0_to_7cm
        self.soil_temperature_7_to_28cm = soil_temperature_7_to_28cm
        self.soil_temperature_28_to_100cm = soil_temperature_28_to_100cm
        self.soil_temperature_100_to_255cm = soil_temperature_100_to_255cm
        self.soil_moisture_0_to_7cm = soil_moisture_0_to_7cm
        self.soil_moisture_7_to_28cm = soil_moisture_7_to_28cm
        self.soil_moisture_28_to_100cm = soil_moisture_28_to_100cm
        self.soil_moisture_100_to_255cm = soil_moisture_100_to_255cm


class HourlyUnitsModel():

    def __init__(self,
                 time: Optional[list[str]] = None,
                 temperature_2m: Optional[list[float]] = None,
                 relativehumidity_2m: Optional[list[int]] = None,
                 dewpoint_2m: Optional[list[float]] = None,
                 apparent_temperature: Optional[list[float]] = None,
                 pressure_msl: Optional[list[float]] = None,
                 surface_pressure: Optional[list[float]] = None,
                 precipitation: Optional[list[float]] = None,
                 rain: Optional[list[float]] = None,
                 snowfall: Optional[list[float]] = None,
                 cloudcover: Optional[list[int]] = None,
                 cloudcover_low: Optional[list[int]] = None,
                 cloudcover_mid: Optional[list[int]] = None,
                 cloudcover_high: Optional[list[int]] = None,
                 shortwave_radiation: Optional[list[int]] = None,
                 direct_radiation: Optional[list[int]] = None,
                 diffuse_radiation: Optional[list[int]] = None,
                 direct_normal_irradiance: Optional[list[float]] = None,
                 windspeed_10m: Optional[list[float]] = None,
                 windspeed_100m: Optional[list[float]] = None,
                 winddirection_10m: Optional[list[int]] = None,
                 winddirection_100m: Optional[list[int]] = None,
                 windgusts_10m: Optional[list[float]] = None,
                 et0_fao_evapotranspiration: Optional[list[float]] = None,
                 vapor_pressure_deficit: Optional[list[float]] = None,
                 soil_temperature_0_to_7cm: Optional[list[float]] = None,
                 soil_temperature_7_to_28cm: Optional[list[float]] = None,
                 soil_temperature_28_to_100cm: Optional[list[float]] = None,
                 soil_temperature_100_to_255cm: Optional[list[float]] = None,
                 soil_moisture_0_to_7cm: Optional[list[float]] = None,
                 soil_moisture_7_to_28cm: Optional[list[float]] = None,
                 soil_moisture_28_to_100cm: Optional[list[float]] = None,
                 soil_moisture_100_to_255cm: Optional[list[float]] = None
                 ):

        self.time = time
        self.temperature_2m = temperature_2m
        self.relativehumidity_2m = relativehumidity_2m
        self.dewpoint_2m = dewpoint_2m
        self.apparent_temperature = apparent_temperature
        self.pressure_msl = pressure_msl
        self.surface_pressure = surface_pressure
        self.precipitation = precipitation
        self.rain = rain
        self.snowfall = snowfall
        self.cloudcover = cloudcover
        self.cloudcover_low = cloudcover_low
        self.cloudcover_mid = cloudcover_mid
        self.cloudcover_high = cloudcover_high
        self.shortwave_radiation = shortwave_radiation
        self.direct_radiation = direct_radiation
        self.diffuse_radiation = diffuse_radiation
        self.direct_normal_irradiance = direct_normal_irradiance
        self.windspeed_10m = windspeed_10m
        self.windspeed_100m = windspeed_100m
        self.winddirection_10m = winddirection_10m
        self.winddirection_100m = winddirection_100m
        self.windgusts_10m = windgusts_10m
        self.et0_fao_evapotranspiration = et0_fao_evapotranspiration
        self.vapor_pressure_deficit = vapor_pressure_deficit
        self.soil_temperature_0_to_7cm = soil_temperature_0_to_7cm
        self.soil_temperature_7_to_28cm = soil_temperature_7_to_28cm
        self.soil_temperature_28_to_100cm = soil_temperature_28_to_100cm
        self.soil_temperature_100_to_255cm = soil_temperature_100_to_255cm
        self.soil_moisture_0_to_7cm = soil_moisture_0_to_7cm
        self.soil_moisture_7_to_28cm = soil_moisture_7_to_28cm
        self.soil_moisture_28_to_100cm = soil_moisture_28_to_100cm
        self.soil_moisture_100_to_255cm = soil_moisture_100_to_255cm


class ResponseModel():

    def __init__(self,
                 latitude: Optional[float] = None,
                 longitude: Optional[float] = None,
                 generationtime_ms: Optional[float] = None,
                 utc_offset_seconds: Optional[int] = None,
                 timezone: Optional[str] = None,
                 timezone_abbreviation: Optional[str] = None,
                 elevation: Optional[int] = None,
                 hourly_units: Optional[HourlyUnitsModel] = None,
                 hourly: Optional[HourlyModel] = None
                 ):

        self.latitude = latitude
        self.longitude = longitude
        self.generationtime_ms = generationtime_ms
        self.utc_offset_seconds = utc_offset_seconds
        self.timezone = timezone
        self.timezone_abbreviation = timezone_abbreviation
        self.elevation = elevation
        self.hourly_units = [] if not hourly_units else HourlyUnitsModel(
            **hourly_units)
        self.hourly = [] if not hourly else HourlyModel(**hourly)
