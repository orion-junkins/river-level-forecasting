import pytest

from rlf.forecasting.data_fetching_utilities.weather_provider.open_meteo.open_meteo_adapter import OpenMeteoAdapter
from rlf.forecasting.data_fetching_utilities.weather_provider.api.api import RequestBuilder
from rlf.forecasting.data_fetching_utilities.weather_provider.api.models import Response


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
def open_meteo_adapter():
    return OpenMeteoAdapter(longitude=fake_longitude, latitude=fake_latitude, start_date=fake_start_date, end_date=fake_end_date)


@pytest.fixture
def request_builder(open_meteo_adapter):
    return RequestBuilder(open_meteo_adapter)


def test_get_returns_response(request_builder):
    response = request_builder.get()
    assert isinstance(response, Response)


def test_get_returns_response_with_status_code(request_builder):
    response = request_builder.get()
    assert response.status_code is not None


def test_get_returns_response_with_content(request_builder):
    response = request_builder.get()
    assert response.data is not None


def test_api_adapter_can_get_payload(open_meteo_adapter):
    payload = open_meteo_adapter.get_payload()
    assert payload is not None
    assert isinstance(payload, dict)
