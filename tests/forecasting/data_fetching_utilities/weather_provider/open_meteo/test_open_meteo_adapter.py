import pytest
from rlf.forecasting.data_fetching_utilities.coordinate import Coordinate

from rlf.forecasting.data_fetching_utilities.weather_provider.open_meteo.open_meteo_adapter import OpenMeteoAdapter


@pytest.fixture
def fake_latitude() -> float:
    return 30.0


@pytest.fixture
def fake_longitude() -> float:
    return 110.0


@pytest.fixture
def fake_coordinate(fake_longitude, fake_latitude) -> Coordinate:
    return Coordinate(lon=fake_longitude, lat=fake_latitude)


@pytest.fixture
def fake_start_date() -> str:
    return "2022-09-13"


@pytest.fixture
def fake_end_date() -> str:
    return "2022-09-14"


@pytest.fixture
def weather_api_parameters() -> OpenMeteoAdapter:
    return OpenMeteoAdapter()


@pytest.mark.slow
def test_get_historical_columns_subset(fake_coordinate, fake_start_date, fake_end_date):
    adapter = OpenMeteoAdapter()
    response = adapter.get_historical(coordinate=fake_coordinate, start_date=fake_start_date, end_date=fake_end_date, columns=["temperature_2m"])
    assert response.status_code == 200
    assert pytest.approx(110.0) == response.data["longitude"]
    assert pytest.approx(30.0) == response.data["latitude"]
    assert len(response.data["hourly"]) == 2  # Time column and 'temperature_2m' column
    assert len(response.data["hourly_units"]) == 2


@pytest.mark.slow
def test_get_current_columns_subset(fake_coordinate):
    adapter = OpenMeteoAdapter()
    response = adapter.get_current(coordinate=fake_coordinate, past_days=1, forecast_days=1, columns=["temperature_2m"])
    assert response.status_code == 200
    assert pytest.approx(110.0) == response.data["longitude"]
    assert pytest.approx(30.0) == response.data["latitude"]
    assert len(response.data["hourly"]) == 2  # Time column and 'temperature_2m' column
    assert len(response.data["hourly_units"]) == 2
