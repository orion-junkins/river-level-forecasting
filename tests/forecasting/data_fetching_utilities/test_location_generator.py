import pytest

from rlf.forecasting.data_fetching_utilities.coordinate import Coordinate
from rlf.forecasting.data_fetching_utilities.location_generator import \
    LocationGenerator

FLOAT_COMPARISON_PRECISION = 0.0001


@pytest.fixture
def bottom_left():
    return Coordinate(lon=0.2, lat=0.00001)


@pytest.fixture
def top_right():
    return Coordinate(lon=0.6, lat=0.4)


@pytest.fixture
def location_generator(bottom_left, top_right):
    return LocationGenerator(bottom_left, top_right)


def test_negative_separation_degrees_is_invalid(bottom_left, top_right):
    separation_degrees = -1
    with pytest.raises(ValueError):
        LocationGenerator(bottom_left, top_right, separation_degrees=separation_degrees)


def test_left_lon_must_be_less_than_right_lon(bottom_left):
    top_right = Coordinate(lon=-1, lat=0.4)
    with pytest.raises(ValueError):
        LocationGenerator(bottom_left, top_right)


def test_bottom_lat_must_be_less_than_top_lat(bottom_left):
    top_right = Coordinate(lon=2, lat=-1)
    with pytest.raises(ValueError):
        LocationGenerator(bottom_left, top_right)


def test_bottom_left(location_generator):
    assert (location_generator.bottom_left == Coordinate(lon=0,  lat=0))


def test_bottom_right(location_generator):
    assert (location_generator.bottom_right == Coordinate(lon=0.0, lat=0.5))


def test_top_left(location_generator):
    assert (location_generator.top_left == Coordinate(lon=0.75,  lat=0.0))


def test_top_right(location_generator):
    assert (location_generator.top_right == Coordinate(lon=0.75, lat=0.5))


def test_lat_excess(location_generator):
    expected_result = 0.0
    assert (abs(location_generator.lat_excess - expected_result) < FLOAT_COMPARISON_PRECISION)


def test_lon_excess(location_generator):
    expected_result = 0.0
    assert (abs(location_generator.lon_excess - expected_result) < FLOAT_COMPARISON_PRECISION)


def test_coordinates(location_generator):
    expected_result = [Coordinate(lon=0.0, lat=0.0),
                       Coordinate(lon=0.25, lat=0.0),
                       Coordinate(lon=0.5, lat=0.0),
                       Coordinate(lon=0.75, lat=0.0),
                       Coordinate(lon=0.0, lat=0.25),
                       Coordinate(lon=0.25, lat=0.25),
                       Coordinate(lon=0.5, lat=0.25),
                       Coordinate(lon=0.75, lat=0.25),
                       Coordinate(lon=0.0, lat=0.5),
                       Coordinate(lon=0.25, lat=0.5),
                       Coordinate(lon=0.5, lat=0.5),
                       Coordinate(lon=0.75, lat=0.5)]
    for (expected, generated) in zip(expected_result, location_generator.coordinates):
        assert (abs(expected.lat - generated.lat) < FLOAT_COMPARISON_PRECISION)
        assert (abs(expected.lon - generated.lon) < FLOAT_COMPARISON_PRECISION)
