class Hourly():

    def __init__(self):
        self.variables: list[str] = []

    def get_variables(self) -> list[str]:
        return self.variables

    def set_temperature_2m(self) -> None:
        self.variables.append("temperature_2m")

    def set_relativehumidity_2m(self) -> None:
        self.variables.append("relativehumidity_2m")

    def set_dewpoint_2m(self) -> None:
        self.variables.append("dewpoint_2m")

    def set_apparent_temperature(self) -> None:
        self.variables.append("apparent_temperature")

    def set_pressure_msl(self) -> None:
        self.variables.append("pressure_msl")

    def set_surface_pressure(self) -> None:
        self.variables.append("surface_pressure")

    def set_precipitation(self) -> None:
        self.variables.append("precipitation")

    def set_rain(self) -> None:
        self.variables.append("rain")

    def set_snowfall(self) -> None:
        self.variables.append("snowfall")

    def set_cloudcover(self) -> None:
        self.variables.append("cloudcover")

    def set_cloudcover_low(self) -> None:
        self.variables.append("cloudcover_low")

    def set_cloudcover_mid(self) -> None:
        self.variables.append("cloudcover_mid")

    def set_cloudcover_high(self) -> None:
        self.variables.append("cloudcover_high")

    def set_shortwave_radiation(self) -> None:
        self.variables.append("shortwave_radiation")

    def set_direct_radiation(self) -> None:
        self.variables.append("direct_radiation")

    def set_diffuse_radiation(self) -> None:
        self.variables.append("diffuse_radiation")

    def set_direct_normal_irradiance(self) -> None:
        self.variables.append("direct_normal_irradiance")

    def set_windspeed_10m(self) -> None:
        self.variables.append("windspeed_10m")

    def set_windspeed_100m(self) -> None:
        self.variables.append("windspeed_100m")

    def set_winddirection_10m(self) -> None:
        self.variables.append("winddirection_10m")

    def set_winddirection_100m(self) -> None:
        self.variables.append("winddirection_100m")

    def set_windgusts_10m(self) -> None:
        self.variables.append("windgusts_10m")

    def set_et0_fao_evapotranspiration(self) -> None:
        self.variables.append("et0_fao_evapotranspiration")

    def set_vapor_pressure_deficit(self) -> None:
        self.variables.append("vapor_pressure_deficit")

    def set_soil_temperature_0_to_7cm(self) -> None:
        self.variables.append("soil_temperature_0_to_7cm")

    def set_soil_temperature_7_to_28cm(self) -> None:
        self.variables.append("soil_temperature_7_to_28cm")

    def set_soil_temperature_28_to_100cm(self) -> None:
        self.variables.append(
            "soil_temperature_28_to_100cm")

    def set_soil_temperature_100_to_255cm(self) -> None:
        self.variables.append(
            "soil_temperature_100_to_255cm")

    def set_soil_moisture_0_to_7cm(self) -> None:
        self.variables.append("soil_moisture_0_to_7cm")

    def set_soil_moisture_7_to_28cm(self) -> None:
        self.variables.append("soil_moisture_7_to_28cm")

    def set_soil_moisture_28_to_100cm(self) -> None:
        self.variables.append("soil_moisture_28_to_100cm")

    def set_soil_moisture_100_to_255cm(self) -> None:
        self.variables.append("soil_moisture_100_to_255cm")

    def set_all(self) -> None:
        self.set_temperature_2m()
        self.set_relativehumidity_2m()
        self.set_dewpoint_2m()
        self.set_apparent_temperature()
        self.set_pressure_msl()
        self.set_surface_pressure()
        self.set_precipitation()
        self.set_rain()
        self.set_snowfall()
        self.set_cloudcover()
        self.set_cloudcover_low()
        self.set_cloudcover_mid()
        self.set_cloudcover_high()
        self.set_shortwave_radiation()
        self.set_direct_radiation()
        self.set_diffuse_radiation()
        self.set_direct_normal_irradiance()
        self.set_windspeed_10m()
        self.set_windspeed_100m()
        self.set_winddirection_10m()
        self.set_winddirection_100m()
        self.set_windgusts_10m()
        self.set_et0_fao_evapotranspiration()
        self.set_vapor_pressure_deficit()
        self.set_soil_temperature_0_to_7cm()
        self.set_soil_temperature_7_to_28cm()
        self.set_soil_temperature_28_to_100cm()
        self.set_soil_temperature_100_to_255cm()
        self.set_soil_moisture_0_to_7cm()
        self.set_soil_moisture_7_to_28cm()
        self.set_soil_moisture_28_to_100cm()
        self.set_soil_moisture_100_to_255cm()
