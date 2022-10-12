class WeatherAPIHourlyParameter():

    def __init__(self):
        """Control valid weather_variables for weather API hourly parameter.

        returns: list of valid weather_variables to build query string for the hourly parameter."""
        self.weather_variables: list[str] = []

    def get_query_string(self) -> str:
        """Get query string for hourly parameter.

        returns: query string for hourly parameter."""
        return "hourly=" + ",".join(self.weather_variables)

    def get_weather_variable_names(self) -> list[str]:
        """Get list of variable names to be included in query hourly parameter.

        returns: list of variable names."""
        return self.weather_variables

    def select_temperature_2m(self) -> None:
        self.weather_variables.append("temperature_2m")

    def select_relativehumidity_2m(self) -> None:
        self.weather_variables.append("relativehumidity_2m")

    def select_dewpoint_2m(self) -> None:
        self.weather_variables.append("dewpoint_2m")

    def select_apparent_temperature(self) -> None:
        self.weather_variables.append("apparent_temperature")

    def select_pressure_msl(self) -> None:
        self.weather_variables.append("pressure_msl")

    def select_surface_pressure(self) -> None:
        self.weather_variables.append("surface_pressure")

    def select_precipitation(self) -> None:
        self.weather_variables.append("precipitation")

    def select_rain(self) -> None:
        self.weather_variables.append("rain")

    def select_snowfall(self) -> None:
        self.weather_variables.append("snowfall")

    def select_cloudcover(self) -> None:
        self.weather_variables.append("cloudcover")

    def select_cloudcover_low(self) -> None:
        self.weather_variables.append("cloudcover_low")

    def select_cloudcover_mid(self) -> None:
        self.weather_variables.append("cloudcover_mid")

    def select_cloudcover_high(self) -> None:
        self.weather_variables.append("cloudcover_high")

    def select_shortwave_radiation(self) -> None:
        self.weather_variables.append("shortwave_radiation")

    def select_direct_radiation(self) -> None:
        self.weather_variables.append("direct_radiation")

    def select_diffuse_radiation(self) -> None:
        self.weather_variables.append("diffuse_radiation")

    def select_direct_normal_irradiance(self) -> None:
        self.weather_variables.append("direct_normal_irradiance")

    def select_windspeed_10m(self) -> None:
        self.weather_variables.append("windspeed_10m")

    def select_windspeed_100m(self) -> None:
        self.weather_variables.append("windspeed_100m")

    def select_winddirection_10m(self) -> None:
        self.weather_variables.append("winddirection_10m")

    def select_winddirection_100m(self) -> None:
        self.weather_variables.append("winddirection_100m")

    def select_windgusts_10m(self) -> None:
        self.weather_variables.append("windgusts_10m")

    def select_et0_fao_evapotranspiration(self) -> None:
        self.weather_variables.append("et0_fao_evapotranspiration")

    def select_vapor_pressure_deficit(self) -> None:
        self.weather_variables.append("vapor_pressure_defiselect_cit")

    def select_soil_temperature_0_to_7cm(self) -> None:
        self.weather_variables.append("soil_temperature_0_to_7cm")

    def select_soil_temperature_7_to_28cm(self) -> None:
        self.weather_variables.append("soil_temperature_7_to_28cm")

    def select_soil_temperature_28_to_100cm(self) -> None:
        self.weather_variables.append(
            "soil_temperature_28_to_100cm")

    def select_soil_temperature_100_to_255cm(self) -> None:
        self.weather_variables.append(
            "soil_temperature_100_to_255cm")

    def select_soil_moisture_0_to_7cm(self) -> None:
        self.weather_variables.append("soil_moisture_0_to_7cm")

    def select_soil_moisture_7_to_28cm(self) -> None:
        self.weather_variables.append("soil_moisture_7_to_28cm")

    def select_soil_moisture_28_to_100cm(self) -> None:
        self.weather_variables.append("soil_moisture_28_to_100cm")

    def select_soil_moisture_100_to_255cm(self) -> None:
        self.weather_variables.append("soil_moisture_100_to_255cm")

    def select_all(self) -> None:
        self.select_temperature_2m()
        self.select_relativehumidity_2m()
        self.select_dewpoint_2m()
        self.select_apparent_temperature()
        self.select_pressure_msl()
        self.select_surface_pressure()
        self.select_precipitation()
        self.select_rain()
        self.select_snowfall()
        self.select_cloudcover()
        self.select_cloudcover_low()
        self.select_cloudcover_mid()
        self.select_cloudcover_high()
        self.select_shortwave_radiation()
        self.select_direct_radiation()
        self.select_diffuse_radiation()
        self.select_direct_normal_irradiance()
        self.select_windspeed_10m()
        self.select_windspeed_100m()
        self.select_winddirection_10m()
        self.select_winddirection_100m()
        self.select_windgusts_10m()
        self.select_et0_fao_evapotranspiration()
        self.select_vapor_pressure_deficit()
        self.select_soil_temperature_0_to_7cm()
        self.select_soil_temperature_7_to_28cm()
        self.select_soil_temperature_28_to_100cm()
        self.select_soil_temperature_100_to_255cm()
        self.select_soil_moisture_0_to_7cm()
        self.select_soil_moisture_7_to_28cm()
        self.select_soil_moisture_28_to_100cm()
        self.select_soil_moisture_100_to_255cm()
