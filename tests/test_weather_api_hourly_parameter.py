import pytest

from forecasting.data_fetching_utilities.historical_weather.weather_api_hourly_parameter import WeatherAPIHourlyParameter


class TestWeatherAPIHourlyParameter():

    @pytest.fixture
    def weather_api_hourly_parameter(self):
        return WeatherAPIHourlyParameter()

    def test_has_weather_variables_attribute(self, weather_api_hourly_parameter):
        assert hasattr(weather_api_hourly_parameter, "weather_variables")

    def test_weather_variables_attribute_is_list(self, weather_api_hourly_parameter):
        assert isinstance(weather_api_hourly_parameter.weather_variables, list)

    def test_weather_variables_attribute_is_empty_list(self, weather_api_hourly_parameter):
        assert weather_api_hourly_parameter.weather_variables == []

    def test_get_weather_variable_names_empty(self, weather_api_hourly_parameter):
        assert weather_api_hourly_parameter.get_weather_variable_names() == []

    def test_get_weather_variable_names_not_empty(self):
        weather_api_hourly_parameter = WeatherAPIHourlyParameter()
        weather_api_hourly_parameter.temperature_2m()
        assert weather_api_hourly_parameter.get_weather_variable_names() == [
            "temperature_2m"]
