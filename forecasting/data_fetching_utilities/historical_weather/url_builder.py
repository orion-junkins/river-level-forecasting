from forecasting.data_fetching_utilities.historical_weather.weather_api_parameters import WeatherAPIParameters


class UrlBuilder():

    def __init__(self, protocol: str = None, host: str = None, path: str = None, parameters: WeatherAPIParameters = WeatherAPIParameters()):
        self.protocol = protocol
        self.host = host
        self.path = path
        self.parameters = parameters

    def set_protocol(self, protocol: str) -> None:
        self.protocol = protocol

    def get_protocol(self) -> str:
        return self.protocol

    def set_host(self, host: str) -> None:
        self.host = host

    def get_host(self) -> str:
        return self.host

    def set_path(self, path: str) -> None:
        self.path = path

    def get_path(self) -> str:
        return self.path

    def set_parameters(self, parameters: WeatherAPIParameters) -> None:
        self.parameters = parameters

    def get_parameters(self) -> WeatherAPIParameters:
        return self.parameters

    def build(self) -> str:
        return self.protocol + "://" + self.host + self.path + "?" + self.parameters.build_query_string()
