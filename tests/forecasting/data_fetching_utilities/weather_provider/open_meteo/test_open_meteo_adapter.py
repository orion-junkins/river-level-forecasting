import pytest
from rlf.forecasting.data_fetching_utilities.coordinate import Coordinate

from rlf.forecasting.data_fetching_utilities.weather_provider.open_meteo.open_meteo_adapter import OpenMeteoAdapter


@pytest.fixture
def fake_latitude() -> float:
    return 30.0


@pytest.fixture
def fake_longitude() -> float:
    return 30.0


@pytest.fixture
def fake_coordinate(fake_longitude, fake_latitude) -> Coordinate:
    return Coordinate(lon=fake_longitude, lat=fake_latitude)


@pytest.fixture
def fake_start_date() -> str:
    return "2022-08-14"


@pytest.fixture
def fake_end_date() -> str:
    return "2022-09-14"


@pytest.fixture
def weather_api_parameters() -> OpenMeteoAdapter:
    return OpenMeteoAdapter()


@pytest.mark.slow
def test_get_historical_returns_response_with_api_data(fake_coordinate, fake_start_date, fake_end_date):
    adapter = OpenMeteoAdapter()
    response = adapter.get_historical(coordinate=fake_coordinate, start_date=fake_start_date, end_date=fake_end_date)
    assert response.status_code == 200
    assert response.data["longitude"] == 30.0
    assert response.data["latitude"] == 30.0
    assert response.data["hourly"]["temperature_2m"][0] == 23.6


@pytest.mark.slow
def test_get_historical_columns_subset(fake_coordinate, fake_start_date, fake_end_date):
    adapter = OpenMeteoAdapter()
    response = adapter.get_historical(coordinate=fake_coordinate, start_date=fake_start_date, end_date=fake_end_date, columns=["temperature_2m"])
    assert response.status_code == 200
    assert len(response.data["hourly"]) == 2  # Time column and 'temperature_2m' column
    assert len(response.data["hourly_units"]) == 2


@pytest.mark.slow
def test_get_current_returns_response_with_api_data(fake_coordinate):
    adapter = OpenMeteoAdapter()
    response = adapter.get_current(coordinate=fake_coordinate, past_days=1, forecast_days=1)
    assert response.status_code == 200
    assert response.data["longitude"] == 30.0
    assert response.data["latitude"] == 30.0
    assert len(response.data['hourly']["temperature_2m"]) == 48  # 2 days of hourly data


@pytest.mark.slow
def test_get_current_columns_subset(fake_coordinate):
    adapter = OpenMeteoAdapter()
    response = adapter.get_current(coordinate=fake_coordinate, past_days=1, forecast_days=1, columns=["temperature_2m"])
    assert response.status_code == 200
    assert len(response.data["hourly"]) == 2  # Time column and 'temperature_2m' column
    assert len(response.data["hourly_units"]) == 2
