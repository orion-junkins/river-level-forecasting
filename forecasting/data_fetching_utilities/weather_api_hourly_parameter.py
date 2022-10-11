class Variable():
    def __init__(self, name: str):
        self.name = name

    def get_name(self) -> str:
        return self.name

    def set_name(self, name: str) -> None:
        self.name = name


class WeatherAPIHourlyParameter():

    def __init__(self):
        """Control valid weather_variables for weather API hourly parameter.

        returns: list of valid weather_variables to build query string for the hourly parameter."""
        self.weather_variables: list[Variable] = []

    def get_weather_variable_names(self) -> list[str]:
        """Get list of variable names to be included in query hourly parameter.

        returns: list of variable names."""
        return [variable.get_name() for variable in self.weather_variables]

    def temperature_2m(self) -> None:
        self.weather_variables.append(Variable("temperature_2m"))

    def relativehumidity_2m(self) -> None:
        self.weather_variables.append(Variable("relativehumidity_2m"))

    def dewpoint_2m(self) -> None:
        self.weather_variables.append(Variable("dewpoint_2m"))

    def apparent_temperature(self) -> None:
        self.weather_variables.append(Variable("apparent_temperature"))

    def pressure_msl(self) -> None:
        self.weather_variables.append(Variable("pressure_msl"))

    def surface_pressure(self) -> None:
        self.weather_variables.append(Variable("surface_pressure"))

    def precipitation(self) -> None:
        self.weather_variables.append(Variable("precipitation"))

    def rain(self) -> None:
        self.weather_variables.append(Variable("rain"))

    def snowfall(self) -> None:
        self.weather_variables.append(Variable("snowfall"))

    def cloudcover(self) -> None:
        self.weather_variables.append(Variable("cloudcover"))

    def cloudcover_low(self) -> None:
        self.weather_variables.append(Variable("cloudcover_low"))

    def cloudcover_mid(self) -> None:
        self.weather_variables.append(Variable("cloudcover_mid"))

    def cloudcover_high(self) -> None:
        self.weather_variables.append(Variable("cloudcover_high"))

    def shortwave_radiation(self) -> None:
        self.weather_variables.append(Variable("shortwave_radiation"))

    def direct_radiation(self) -> None:
        self.weather_variables.append(Variable("direct_radiation"))

    def diffuse_radiation(self) -> None:
        self.weather_variables.append(Variable("diffuse_radiation"))

    def direct_normal_irradiance(self) -> None:
        self.weather_variables.append(Variable("direct_normal_irradiance"))

    def windspeed_10m(self) -> None:
        self.weather_variables.append(Variable("windspeed_10m"))

    def windspeed_100m(self) -> None:
        self.weather_variables.append(Variable("windspeed_100m"))

    def winddirection_10m(self) -> None:
        self.weather_variables.append(Variable("winddirection_10m"))

    def winddirection_100m(self) -> None:
        self.weather_variables.append(Variable("winddirection_100m"))

    def windgusts_10m(self) -> None:
        self.weather_variables.append(Variable("windgusts_10m"))

    def et0_fao_evapotranspiration(self) -> None:
        self.weather_variables.append(Variable("et0_fao_evapotranspiration"))

    def vapor_pressure_deficit(self) -> None:
        self.weather_variables.append(Variable("vapor_pressure_deficit"))

    def soil_temperature_0_to_7cm(self) -> None:
        self.weather_variables.append(Variable("soil_temperature_0_to_7cm"))

    def soil_temperature_7_to_28cm(self) -> None:
        self.weather_variables.append(Variable("soil_temperature_7_to_28cm"))

    def soil_temperature_28_to_100cm(self) -> None:
        self.weather_variables.append(
            Variable("soil_temperature_28_to_100cm"))

    def soil_temperature_100_to_255cm(self) -> None:
        self.weather_variables.append(
            Variable("soil_temperature_100_to_255cm"))

    def soil_moisture_0_to_7cm(self) -> None:
        self.weather_variables.append(Variable("soil_moisture_0_to_7cm"))

    def soil_moisture_7_to_28cm(self) -> None:
        self.weather_variables.append(Variable("soil_moisture_7_to_28cm"))

    def soil_moisture_28_to_100cm(self) -> None:
        self.weather_variables.append(Variable("soil_moisture_28_to_100cm"))

    def soil_moisture_100_to_255cm(self) -> None:
        self.weather_variables.append(Variable("soil_moisture_100_to_255cm"))
