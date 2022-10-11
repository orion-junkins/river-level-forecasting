class WeatherAPIHourlyParameter():

    def __init__(self):
        """Control valid weather_variables for weather API hourly parameter.

        returns: list of valid weather_variables to build query string for the hourly parameter."""
        self.weather_variables: list[str] = []

    def get_weather_variable_names(self) -> list[str]:
        """Get list of variable names to be included in query hourly parameter.

        returns: list of variable names."""
        return self.weather_variables

    def temperature_2m(self) -> None:
        self.weather_variables.append("temperature_2m")

    def relativehumidity_2m(self) -> None:
        self.weather_variables.append("relativehumidity_2m")

    def dewpoint_2m(self) -> None:
        self.weather_variables.append("dewpoint_2m")

    def apparent_temperature(self) -> None:
        self.weather_variables.append("apparent_temperature")

    def pressure_msl(self) -> None:
        self.weather_variables.append("pressure_msl")

    def surface_pressure(self) -> None:
        self.weather_variables.append("surface_pressure")

    def precipitation(self) -> None:
        self.weather_variables.append("precipitation")

    def rain(self) -> None:
        self.weather_variables.append("rain")

    def snowfall(self) -> None:
        self.weather_variables.append("snowfall")

    def cloudcover(self) -> None:
        self.weather_variables.append("cloudcover")

    def cloudcover_low(self) -> None:
        self.weather_variables.append("cloudcover_low")

    def cloudcover_mid(self) -> None:
        self.weather_variables.append("cloudcover_mid")

    def cloudcover_high(self) -> None:
        self.weather_variables.append("cloudcover_high")

    def shortwave_radiation(self) -> None:
        self.weather_variables.append("shortwave_radiation")

    def direct_radiation(self) -> None:
        self.weather_variables.append("direct_radiation")

    def diffuse_radiation(self) -> None:
        self.weather_variables.append("diffuse_radiation")

    def direct_normal_irradiance(self) -> None:
        self.weather_variables.append("direct_normal_irradiance")

    def windspeed_10m(self) -> None:
        self.weather_variables.append("windspeed_10m")

    def windspeed_100m(self) -> None:
        self.weather_variables.append("windspeed_100m")

    def winddirection_10m(self) -> None:
        self.weather_variables.append("winddirection_10m")

    def winddirection_100m(self) -> None:
        self.weather_variables.append("winddirection_100m")

    def windgusts_10m(self) -> None:
        self.weather_variables.append("windgusts_10m")

    def et0_fao_evapotranspiration(self) -> None:
        self.weather_variables.append("et0_fao_evapotranspiration")

    def vapor_pressure_deficit(self) -> None:
        self.weather_variables.append("vapor_pressure_deficit")

    def soil_temperature_0_to_7cm(self) -> None:
        self.weather_variables.append("soil_temperature_0_to_7cm")

    def soil_temperature_7_to_28cm(self) -> None:
        self.weather_variables.append("soil_temperature_7_to_28cm")

    def soil_temperature_28_to_100cm(self) -> None:
        self.weather_variables.append(
            "soil_temperature_28_to_100cm")

    def soil_temperature_100_to_255cm(self) -> None:
        self.weather_variables.append(
            "soil_temperature_100_to_255cm")

    def soil_moisture_0_to_7cm(self) -> None:
        self.weather_variables.append("soil_moisture_0_to_7cm")

    def soil_moisture_7_to_28cm(self) -> None:
        self.weather_variables.append("soil_moisture_7_to_28cm")

    def soil_moisture_28_to_100cm(self) -> None:
        self.weather_variables.append("soil_moisture_28_to_100cm")

    def soil_moisture_100_to_255cm(self) -> None:
        self.weather_variables.append("soil_moisture_100_to_255cm")
