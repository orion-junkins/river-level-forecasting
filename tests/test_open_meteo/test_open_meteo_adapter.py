import pytest
from forecasting.data_fetching_utilities.open_meteo.open_meteo_adapter import OpenMeteoAdapter


class TestOpenMeteoAdapter():

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
    def weather_api_parameters(self) -> OpenMeteoAdapter:
        return OpenMeteoAdapter()

    def test_has_base_url_attributes(self, weather_api_parameters):
        assert hasattr(weather_api_parameters, "protocol")
        assert hasattr(weather_api_parameters, "hostname")
        assert hasattr(weather_api_parameters, "version")
        assert hasattr(weather_api_parameters, "path")

    def test_has_latitude(self, weather_api_parameters):
        assert (weather_api_parameters.latitude)

    def test_has_longitude(self, weather_api_parameters):
        assert (weather_api_parameters.longitude)

    def test_has_start_date(self, weather_api_parameters):
        assert hasattr(weather_api_parameters, "start_date")

    def test_has_end_date(self, weather_api_parameters):
        assert hasattr(weather_api_parameters, "end_date")

    def test_has_hourly_parameter(self, weather_api_parameters):
        assert hasattr(weather_api_parameters, "hourly_parameters")

    def test_has_get_payload_method(self, weather_api_parameters):
        assert hasattr(weather_api_parameters, "get_payload")

    def test_get_payload_returns_dict(self, weather_api_parameters):
        assert isinstance(weather_api_parameters.get_payload(), dict)

    def test_get_payload_returns_correct_payload(self, weather_api_parameters):
        assert (weather_api_parameters.get_payload() == {
            "protocol": weather_api_parameters.protocol,
            "hostname": weather_api_parameters.hostname,
            "version": weather_api_parameters.version,
            "path": weather_api_parameters.path,
            "parameters": weather_api_parameters.get_parameters()
        })

    def test_min_valid_latitude(self, fake_min_invalid_latitude):
        with pytest.raises(ValueError):
            OpenMeteoAdapter(latitude=fake_min_invalid_latitude)

    def test_set_location(self, fake_latitude, fake_longitude, weather_api_parameters):
        weather_api_parameters.set_location(fake_latitude, fake_longitude)
        assert (weather_api_parameters.latitude == fake_latitude)
        assert (weather_api_parameters.longitude == fake_longitude)

    def test_get_location_returns_tuple(self, weather_api_parameters):
        assert isinstance(weather_api_parameters.get_location(), tuple)

    def test_get_location_returns_location(self, fake_latitude, fake_longitude, weather_api_parameters):
        weather_api_parameters.set_location(fake_latitude, fake_longitude)
        assert (weather_api_parameters.get_location()
                == (fake_latitude, fake_longitude))

    def test_set_start_date(self, fake_start_date, weather_api_parameters):
        weather_api_parameters.set_start_date(fake_start_date)
        assert (weather_api_parameters.get_start_date() == fake_start_date)

    def test_get_start_date(self, fake_start_date):
        weather_api_parameters = OpenMeteoAdapter(
            start_date=fake_start_date)
        assert (weather_api_parameters.get_start_date() == fake_start_date)

    def test_set_end_date(self, fake_end_date, weather_api_parameters):
        weather_api_parameters.set_end_date(fake_end_date)
        assert (weather_api_parameters.get_end_date() == fake_end_date)

    def test_get_end_date(self, fake_end_date, weather_api_parameters):
        weather_api_parameters.end_date = fake_end_date
        assert (weather_api_parameters.get_end_date() == fake_end_date)
