import pytest
from datetime import date
from forecasting.data_fetching_utilities.historical_weather.weather_api_hourly_parameter import WeatherAPIHourlyParameter
from forecasting.data_fetching_utilities.historical_weather.weather_api_parameters import WeatherAPIParameters


class TestWeatherAPIParameters():

    @pytest.fixture
    def current_date(self):
        return date.today().isoformat()

    @pytest.fixture
    def fake_latitude(self) -> float:
        return 55.55

    @pytest.fixture
    def fake_min_invalid_latitude(self) -> float:
        return -100

    @pytest.fixture
    def fake_max_invalid_latitude(self) -> float:
        return 100

    @pytest.fixture
    def fake_min_invalid_longitude(self) -> float:
        return -200

    @pytest.fixture
    def fake_max_invalid_longitude(self) -> float:
        return 200

    @pytest.fixture
    def fake_longitude(self) -> float:
        return 55.55

    @pytest.fixture
    def fake_start_date(self) -> str:
        return "2022-08-14"

    @pytest.fixture
    def fake_end_date(self) -> str:
        return "2022-09-14"

    @pytest.fixture
    def weather_api_parameters(self) -> WeatherAPIParameters:
        return WeatherAPIParameters()

    def test_has_latitude(self, weather_api_parameters):
        assert (weather_api_parameters.latitude)

    def test_has_longitude(self, weather_api_parameters):
        assert (weather_api_parameters.longitude)

    def test_has_start_date(self, weather_api_parameters):
        assert hasattr(weather_api_parameters, "start_date")

    def test_has_end_date(self, weather_api_parameters):
        assert hasattr(weather_api_parameters, "end_date")

    def test_has_hourly_parameter(self, weather_api_parameters):
        assert hasattr(weather_api_parameters, "hourly_parameter")

    def test_min_valid_latitude(self, fake_min_invalid_latitude):
        with pytest.raises(ValueError):
            WeatherAPIParameters(latitude=fake_min_invalid_latitude)

    def test_set_location(self, fake_latitude, fake_longitude, weather_api_parameters):
        weather_api_parameters.set_location(fake_latitude, fake_longitude)
        assert (weather_api_parameters.latitude == fake_latitude)
        assert (weather_api_parameters.longitude == fake_longitude)

    def test_get_location(self, fake_latitude, fake_longitude, weather_api_parameters):
        weather_api_parameters = WeatherAPIParameters(
            fake_latitude, fake_longitude)
        assert (weather_api_parameters.get_location()
                == (fake_latitude, fake_longitude))

    def test_set_start_date(self, current_date, fake_start_date, weather_api_parameters):
        assert (weather_api_parameters.start_date == current_date)
        weather_api_parameters.set_start_date(start_date=fake_start_date)
        assert (weather_api_parameters.start_date == fake_start_date)

    def test_get_start_date(self, fake_start_date):
        weather_api_parameters = WeatherAPIParameters(
            start_date=fake_start_date)
        assert (weather_api_parameters.get_start_date() == fake_start_date)

    def test_set_end_date(self, current_date, fake_end_date, weather_api_parameters):
        assert (weather_api_parameters.end_date == current_date)
        weather_api_parameters.set_end_date(end_date=fake_end_date)
        assert (weather_api_parameters.end_date == fake_end_date)

    def test_get_end_date(self, fake_end_date, weather_api_parameters):
        weather_api_parameters.end_date = fake_end_date
        assert (weather_api_parameters.get_end_date() == fake_end_date)

    def test_get_hourly_parameter_empty(self, weather_api_parameters):
        assert (weather_api_parameters.get_hourly_parameter() == None)

    def test_get_hourly_parameter_not_empty(self, weather_api_parameters):
        weather_api_hourly_parameter = WeatherAPIHourlyParameter()
        weather_api_hourly_parameter.temperature_2m()
        weather_api_parameters.hourly_parameter = weather_api_hourly_parameter
        assert (weather_api_parameters.get_hourly_parameter()
                == ["temperature_2m"])

    def test_get_query_parameters_base(self, fake_latitude, fake_longitude, fake_start_date, fake_end_date, weather_api_parameters):
        weather_api_parameters.set_location(fake_latitude, fake_longitude)
        weather_api_parameters.set_start_date(fake_start_date)
        weather_api_parameters.set_end_date(fake_end_date)
        query_base_string = f"latitude={fake_latitude}&longitude={fake_longitude}&start_date={fake_start_date}&end_date={fake_end_date}"
        assert (weather_api_parameters.get_query_parameters()
                == query_base_string)

    def test_get_query_parameters_with_hourly(self, fake_latitude, fake_longitude, fake_start_date, fake_end_date):

        weather_api_hourly_parameter = WeatherAPIHourlyParameter()
        weather_api_hourly_parameter.temperature_2m()
        weather_api_hourly_parameter.pressure_msl()
        weather_api_hourly_parameter.diffuse_radiation()

        weather_api_parameters = WeatherAPIParameters(
            hourly_parameter=weather_api_hourly_parameter)
        weather_api_parameters.set_location(fake_latitude, fake_longitude)
        weather_api_parameters.set_start_date(fake_start_date)
        weather_api_parameters.set_end_date(fake_end_date)

        query_base_string = f"latitude={fake_latitude}&longitude={fake_longitude}&start_date={fake_start_date}&end_date={fake_end_date}"
        query_hourly_string = "&hourly=temperature_2m,pressure_msl,diffuse_radiation"

        assert (weather_api_parameters.get_query_parameters()
                == query_base_string + query_hourly_string)
