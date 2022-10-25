import pytest

from rlf.forecasting.data_fetching_utilities.weather_provider.open_meteo.open_meteo_adapter import OpenMeteoAdapter


@pytest.fixture
def fake_latitude() -> float:
    return 1.0


@pytest.fixture
def fake_longitude() -> float:
    return 2.0


@pytest.fixture
def fake_start_date() -> str:
    return "2022-08-14"


@pytest.fixture
def fake_end_date() -> str:
    return "2022-09-14"


@pytest.fixture
def weather_api_parameters(fake_latitude, fake_longitude, fake_start_date, fake_end_date) -> OpenMeteoAdapter:
    return OpenMeteoAdapter(longitude=fake_longitude, latitude=fake_latitude, start_date=fake_start_date, end_date=fake_end_date)


def test_get_returns_response_with_api_data():
    adapter = OpenMeteoAdapter(
        longitude=30.0, latitude=30.0, start_date="2022-08-14", end_date="2022-08-14")
    response = adapter.get()
    assert response.status_code == 200
    assert response.data["longitude"] == 30.0
    assert response.data["latitude"] == 30.0
    assert response.data["elevation"] == 162.0
    assert response.data["hourly"]["temperature_2m"][0] == 23.6


def test_get_location_returns_tuple(weather_api_parameters):
    assert isinstance(weather_api_parameters.get_location(), tuple)


def test_get_location_returns_location(fake_latitude, fake_longitude, weather_api_parameters):
    weather_api_parameters.set_location(fake_latitude, fake_longitude)
    assert (weather_api_parameters.get_location()
            == (fake_latitude, fake_longitude))


def test_set_start_date(fake_start_date, weather_api_parameters):
    weather_api_parameters.set_start_date(fake_start_date)
    assert (weather_api_parameters.get_start_date() == fake_start_date)


def test_get_start_date(weather_api_parameters, fake_start_date):
    assert (weather_api_parameters.get_start_date() == fake_start_date)


def test_set_end_date(fake_end_date, weather_api_parameters):
    weather_api_parameters.set_end_date(fake_end_date)
    assert (weather_api_parameters.get_end_date() == fake_end_date)


def test_get_end_date(fake_end_date, weather_api_parameters):
    weather_api_parameters.end_date = fake_end_date
    assert (weather_api_parameters.get_end_date() == fake_end_date)
