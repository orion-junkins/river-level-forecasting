import pytest

from rlf.forecasting.data_fetching_utilities.coordinate import Coordinate
from rlf.forecasting.data_fetching_utilities.location_generator import \
    LocationGenerator

FLOAT_COMPARISON_PRECISION = 0.0001


@pytest.fixture
def bottom_left():
    return Coordinate(lon=0.21, lat=0.0)


@pytest.fixture
def top_right():
    return Coordinate(lon=0.38, lat=0.19)


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
    assert (location_generator.bottom_left == Coordinate(lon=0.2,  lat=0))


def test_bottom_right(location_generator):
    assert (location_generator.bottom_right == Coordinate(lon=0.4, lat=0.0))


def test_top_left(location_generator):
    assert (location_generator.top_left == Coordinate(lon=0.2,  lat=0.2))


def test_top_right(location_generator):
    assert (location_generator.top_right == Coordinate(lon=0.4, lat=0.2))


def test_coordinates(location_generator):
    expected_result = [Coordinate(lon=0.2, lat=0.0),
                       Coordinate(lon=0.3, lat=0.0),
                       Coordinate(lon=0.4, lat=0.0),
                       Coordinate(lon=0.2, lat=0.1),
                       Coordinate(lon=0.3, lat=0.1),
                       Coordinate(lon=0.4, lat=0.1),
                       Coordinate(lon=0.2, lat=0.2),
                       Coordinate(lon=0.3, lat=0.2),
                       Coordinate(lon=0.4, lat=0.2)]
    for (expected, generated) in zip(expected_result, location_generator.coordinates):
        assert (abs(expected.lat - generated.lat) < FLOAT_COMPARISON_PRECISION)
        assert (abs(expected.lon - generated.lon) < FLOAT_COMPARISON_PRECISION)
