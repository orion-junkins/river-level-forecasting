import pytest
from rlf.forecasting.data_fetching_utilities.weather_provider.open_meteo.open_meteo_adapter import OpenMeteoAdapter


class TestOpenMeteoAdapter():

    @pytest.fixture
    def fake_latitude(self) -> float:
        return 1.0

    @pytest.fixture
    def fake_longitude(self) -> float:
        return 2.0

    @pytest.fixture
    def fake_start_date(self) -> str:
        return "2022-08-14"

    @pytest.fixture
    def fake_end_date(self) -> str:
        return "2022-09-14"

    @pytest.fixture
    def weather_api_parameters(self, fake_latitude, fake_longitude, fake_start_date, fake_end_date) -> OpenMeteoAdapter:
        return OpenMeteoAdapter(longitude=fake_longitude, latitude=fake_latitude, start_date=fake_start_date, end_date=fake_end_date)

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

    def test_get_location_returns_tuple(self, weather_api_parameters):
        assert isinstance(weather_api_parameters.get_location(), tuple)

    def test_get_location_returns_location(self, fake_latitude, fake_longitude, weather_api_parameters):
        weather_api_parameters.set_location(fake_latitude, fake_longitude)
        assert (weather_api_parameters.get_location()
                == (fake_latitude, fake_longitude))

    def test_set_start_date(self, fake_start_date, weather_api_parameters):
        weather_api_parameters.set_start_date(fake_start_date)
        assert (weather_api_parameters.get_start_date() == fake_start_date)

    def test_get_start_date(self, weather_api_parameters, fake_start_date):
        assert (weather_api_parameters.get_start_date() == fake_start_date)

    def test_set_end_date(self, fake_end_date, weather_api_parameters):
        weather_api_parameters.set_end_date(fake_end_date)
        assert (weather_api_parameters.get_end_date() == fake_end_date)

    def test_get_end_date(self, fake_end_date, weather_api_parameters):
        weather_api_parameters.end_date = fake_end_date
        assert (weather_api_parameters.get_end_date() == fake_end_date)
