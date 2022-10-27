import pytest

from rlf.forecasting.data_fetching_utilities.coordinate import Coordinate
from rlf.forecasting.data_fetching_utilities.location_generator import \
    LocationGenerator

FLOAT_COMPARISON_PRECISION = 0.0001


@pytest.fixture
def bottom_left():
    return Coordinate(lon=0, lat=0)


@pytest.fixture
def top_right():
    return Coordinate(lon=26, lat=21)


@pytest.fixture
def location_generator(bottom_left, top_right):
    return LocationGenerator(bottom_left, top_right, separation_distance_km=1000)


def test_negative_separation_distance_is_invalid(bottom_left, top_right):
    separation_distance = -1
    with pytest.raises(ValueError):
        LocationGenerator(bottom_left, top_right, separation_distance_km=separation_distance)


def test_left_lon_must_be_less_than_right_lon(bottom_left):
    top_right = Coordinate(lon=-1, lat=21)
    with pytest.raises(ValueError):
        LocationGenerator(bottom_left, top_right)


def test_bottom_lat_must_be_less_than_top_lat(bottom_left):
    top_right = Coordinate(lon=26, lat=-1)
    with pytest.raises(ValueError):
        LocationGenerator(bottom_left, top_right)


def test_bottom_left(location_generator):
    assert (location_generator.bottom_left == Coordinate(lon=0,  lat=0))


def test_bottom_right(location_generator):
    assert (location_generator.bottom_right == Coordinate(lon=26, lat=0))


def test_top_left(location_generator):
    assert (location_generator.top_left == Coordinate(lon=0,  lat=21))


def test_top_right(location_generator):
    assert (location_generator.top_right == Coordinate(lon=26, lat=21))


def test_lat_separation(location_generator):
    expected_result = 7.50737
    assert (abs(location_generator.lat_separation - expected_result) < FLOAT_COMPARISON_PRECISION)


def test_lon_separation(location_generator):
    expected_result = 11.19205
    assert (abs(location_generator.lon_separation - expected_result) < FLOAT_COMPARISON_PRECISION)


def test_lat_excess(location_generator):
    expected_result = 2.99263
    assert (abs(location_generator.lat_excess - expected_result) < FLOAT_COMPARISON_PRECISION)


def test_lon_excess(location_generator):
    expected_result = 1.80795
    assert (abs(location_generator.lon_excess - expected_result) < FLOAT_COMPARISON_PRECISION)


def test_coordinates(location_generator):
    expected_result = [Coordinate(lon=1.80795, lat=2.99263), Coordinate(lon=13.0, lat=2.99263), Coordinate(lon=24.19205, lat=2.99263), Coordinate(lon=1.80795, lat=10.5), Coordinate(lon=13.0, lat=10.5), Coordinate(lon=24.19205, lat=10.5), Coordinate(lon=1.80795, lat=18.00737), Coordinate(lon=13.0, lat=18.00737), Coordinate(lon=24.19205, lat=18.00737)]
    for (expected, generated) in zip(expected_result, location_generator.coordinates):
        assert (abs(expected.lat - generated.lat) < FLOAT_COMPARISON_PRECISION)
        assert (abs(expected.lon - generated.lon) < FLOAT_COMPARISON_PRECISION)
