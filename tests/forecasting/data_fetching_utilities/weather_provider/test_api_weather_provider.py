from typing import Dict, List, Optional, Union

import pytest

from rlf.forecasting.data_fetching_utilities.coordinate import Coordinate
from rlf.forecasting.data_fetching_utilities.weather_provider.api.base_api_adapter import BaseAPIAdapter
from rlf.forecasting.data_fetching_utilities.weather_provider.api.models import Response
from rlf.forecasting.data_fetching_utilities.weather_provider.api_weather_provider import APIWeatherProvider


def fake_response(coordinate, columns: List[str]):
    full_columns: Dict[str, Union[List[str], List[float]]] = {"time": ['2000-01-01T00:00', '2000-01-01T01:00', '2000-01-01T02:00']}
    full_columns.update({col: [1.0, 2.0, 3.0] for col in columns})

    response = Response(status_code=0,
                        url="fake url",
                        message="fake message",
                        headers={"fake": "headers"},
                        data={'latitude': coordinate.lat,
                              'longitude': coordinate.lon,
                              'generationtime_ms': 0.1,
                              'utc_offset_seconds': 0,
                              'timezone': 'GMT',
                              'timezone_abbreviation': 'GMT',
                              'elevation': 123.4,
                              'hourly_units': {'time': 'iso8601', 'temperature_2m': '°C'},
                              'hourly': full_columns})
    return response


class FakeWeatherAPIAdapter(BaseAPIAdapter):

    def __init__(self, columns: List[str] = ["temperature_2m"], expected_request_columns: Optional[List[str]] = None) -> None:
        self._response_columns = columns
        self._expected_request_columns = expected_request_columns

    # mypy wants this signature to match BaseAPIAdapter, but that will couple the test too hard
    def get_current(self, coordinate: Coordinate, **kwargs) -> Response:  # type: ignore[override]
        if self._expected_request_columns:
            assert "columns" in kwargs
            assert kwargs["columns"] is not None
            assert sorted(self._expected_request_columns) == sorted(kwargs["columns"])
        response = fake_response(coordinate=coordinate, columns=self._response_columns)
        return response

    # mypy wants this signature to match BaseAPIAdapter, but that will couple the test too hard
    def get_historical(self, coordinate: Coordinate, **kwargs) -> Response:  # type: ignore[override]
        if self._expected_request_columns:
            assert "columns" in kwargs
            assert kwargs["columns"] is not None
            assert sorted(self._expected_request_columns) == sorted(kwargs["columns"])
        response = fake_response(coordinate=coordinate, columns=self._response_columns)
        return response

    def get_index_parameter(self) -> str:
        return "time"


@pytest.fixture
def fake_weather_api_adapter() -> FakeWeatherAPIAdapter:
    return FakeWeatherAPIAdapter()


@ pytest.fixture
def weather_provider(fake_weather_api_adapter) -> APIWeatherProvider:
    coordinates = [Coordinate(lon=1.0, lat=2.0),
                   Coordinate(lon=3.0, lat=4.0)]
    weather_provider = APIWeatherProvider(
        coordinates=coordinates, api_adapter=fake_weather_api_adapter)
    return weather_provider


def test_fetch_historical_fetches_one_per_location(weather_provider):
    weather_datums = weather_provider.fetch_historical()
    assert len(weather_datums) == len(weather_provider.coordinates)


def test_fetch_historical_returns_expected_datum(weather_provider):
    weather_datums = weather_provider.fetch_historical()
    for weather_datum in weather_datums:
        assert weather_datum.hourly_parameters.index.dtype == "datetime64[ns, UTC]"
        assert list(weather_datum.hourly_parameters.columns) == ["temperature_2m"]
        assert len(weather_datum.hourly_parameters) == 3


def test_fetch_current_fetches_one_per_location(weather_provider):
    weather_datums = weather_provider.fetch_current()
    assert len(weather_datums) == len(weather_provider.coordinates)


def test_fetch_current_returns_expected_datum(weather_provider):
    weather_datums = weather_provider.fetch_current()
    for weather_datum in weather_datums:
        assert weather_datum.hourly_parameters.index.dtype == "datetime64[ns, UTC]"
        assert list(weather_datum.hourly_parameters.columns) == ["temperature_2m"]
        assert len(weather_datum.hourly_parameters) == 3


def test_fetch_historical_returns_remapped_columns():
    coordinates = [Coordinate(lon=1.0, lat=2.0)]
    request_columns = [
        "soil_temperature_0_to_7cm",
        "soil_temperature_7_to_28cm",
        "soil_temperature_28_to_100cm",
        "soil_temperature_100_to_255cm",
        "soil_moisture_0_to_7cm",
        "soil_moisture_7_to_28cm",
        "soil_moisture_28_to_100cm",
        "soil_moisture_100_to_255cm"
    ]
    fake_adapter = FakeWeatherAPIAdapter(request_columns)
    weather_provider = APIWeatherProvider(coordinates=coordinates, api_adapter=fake_adapter)

    datum = weather_provider.fetch_historical()[0]
    actual_columns = sorted(list(datum.hourly_parameters.columns))
    expected_columns = [
        "soil_moisture_level_1",
        "soil_moisture_level_2",
        "soil_moisture_level_3",
        "soil_moisture_level_4",
        "soil_temperature_level_1",
        "soil_temperature_level_2",
        "soil_temperature_level_3",
        "soil_temperature_level_4",
    ]

    assert expected_columns == actual_columns


def test_fetch_current_returns_remapped_columns():
    coordinates = [Coordinate(lon=1.0, lat=2.0)]
    request_columns = [
        "soil_temperature_0_to_10cm",
        "soil_temperature_10_to_40cm",
        "soil_temperature_40_to_100cm",
        "soil_temperature_100_to_200cm",
        "soil_moisture_0_to_10cm",
        "soil_moisture_10_to_40cm",
        "soil_moisture_40_to_100cm",
        "soil_moisture_100_to_200cm",
    ]
    fake_adapter = FakeWeatherAPIAdapter(request_columns)
    weather_provider = APIWeatherProvider(coordinates=coordinates, api_adapter=fake_adapter)

    datum = weather_provider.fetch_current()[0]
    actual_columns = sorted(list(datum.hourly_parameters.columns))
    expected_columns = [
        "soil_moisture_level_1",
        "soil_moisture_level_2",
        "soil_moisture_level_3",
        "soil_moisture_level_4",
        "soil_temperature_level_1",
        "soil_temperature_level_2",
        "soil_temperature_level_3",
        "soil_temperature_level_4",
    ]

    assert expected_columns == actual_columns


def test_fetch_historical_passed_columns_are_remapped():
    coordinates = [Coordinate(lon=1.0, lat=2.0)]
    response_columns = [
        "soil_temperature_0_to_7cm",
        "soil_temperature_7_to_28cm",
        "soil_temperature_28_to_100cm",
        "soil_temperature_100_to_255cm",
        "soil_moisture_0_to_7cm",
        "soil_moisture_7_to_28cm",
        "soil_moisture_28_to_100cm",
        "soil_moisture_100_to_255cm"
    ]
    request_columns = [
        "soil_moisture_level_1",
        "soil_moisture_level_2",
        "soil_moisture_level_3",
        "soil_moisture_level_4",
        "soil_temperature_level_1",
        "soil_temperature_level_2",
        "soil_temperature_level_3",
        "soil_temperature_level_4",
    ]
    fake_adapter = FakeWeatherAPIAdapter(response_columns, response_columns)

    weather_provider = APIWeatherProvider(coordinates=coordinates, api_adapter=fake_adapter)
    datum = weather_provider.fetch_historical(columns=request_columns)[0]

    actual_columns = sorted(list(datum.hourly_parameters.columns))
    expected_columns = [
        "soil_moisture_level_1",
        "soil_moisture_level_2",
        "soil_moisture_level_3",
        "soil_moisture_level_4",
        "soil_temperature_level_1",
        "soil_temperature_level_2",
        "soil_temperature_level_3",
        "soil_temperature_level_4",
    ]

    assert expected_columns == actual_columns


def test_fetch_current_passed_columns_are_remapped():
    coordinates = [Coordinate(lon=1.0, lat=2.0)]
    response_columns = [
        "soil_temperature_0_to_10cm",
        "soil_temperature_10_to_40cm",
        "soil_temperature_40_to_100cm",
        "soil_temperature_100_to_200cm",
        "soil_moisture_0_to_10cm",
        "soil_moisture_10_to_40cm",
        "soil_moisture_40_to_100cm",
        "soil_moisture_100_to_200cm",
    ]
    request_columns = [
        "soil_moisture_level_1",
        "soil_moisture_level_2",
        "soil_moisture_level_3",
        "soil_moisture_level_4",
        "soil_temperature_level_1",
        "soil_temperature_level_2",
        "soil_temperature_level_3",
        "soil_temperature_level_4",
    ]
    fake_adapter = FakeWeatherAPIAdapter(response_columns, response_columns)

    weather_provider = APIWeatherProvider(coordinates=coordinates, api_adapter=fake_adapter)
    datum = weather_provider.fetch_current(columns=request_columns)[0]

    actual_columns = sorted(list(datum.hourly_parameters.columns))
    expected_columns = [
        "soil_moisture_level_1",
        "soil_moisture_level_2",
        "soil_moisture_level_3",
        "soil_moisture_level_4",
        "soil_temperature_level_1",
        "soil_temperature_level_2",
        "soil_temperature_level_3",
        "soil_temperature_level_4",
    ]

    assert expected_columns == actual_columns
