import pytest

from forecasting.data_fetching_utilities.coordinate import Coordinate
from forecasting.data_fetching_utilities.location_generator import \
    LocationGenerator


@pytest.fixture
def bottom_left():
    return Coordinate(0, 0)


@pytest.fixture
def top_right():
    return Coordinate(21, 26)


@pytest.fixture
def location_generator(bottom_left, top_right):
    return LocationGenerator(bottom_left, top_right, separation_distance_km=1000)


def test_negative_separation_distance_is_invalid(bottom_left, top_right):
    separation_distance = -1
    with pytest.raises(ValueError):
        LocationGenerator(bottom_left, top_right, separation_distance_km=separation_distance)


def test_left_lon_must_be_less_than_right_lon(bottom_left):
    top_right = Coordinate(21, -1)
    with pytest.raises(ValueError):
        LocationGenerator(bottom_left, top_right)


def test_bottom_lat_must_be_less_than_top_lat(bottom_left):
    top_right = Coordinate(-1, 26)
    with pytest.raises(ValueError):
        LocationGenerator(bottom_left, top_right)


def test_bottom_left(location_generator):
    assert (location_generator.bottom_left == Coordinate(0, 0))


def test_bottom_right(location_generator):
    assert (location_generator.bottom_right == Coordinate(0, 26))


def test_top_left(location_generator):
    assert (location_generator.top_left == Coordinate(21, 0))


def test_top_right(location_generator):
    assert (location_generator.top_right == Coordinate(21, 26))


def test_lat_separation(location_generator):
    expected_result = 7.5073669164334405
    assert (location_generator.lat_separation == expected_result)


def test_lon_separation(location_generator):
    expected_result = 11.192053240167708
    location_generator.save_to_kml()
    assert (location_generator.lon_separation == expected_result)


def test_lat_excess(location_generator):
    expected_result = 2.9926330835665595
    assert (location_generator.lat_excess == expected_result)


def test_lon_excess(location_generator):
    expected_result = 1.8079467598322925
    location_generator.save_to_kml()
    assert (location_generator.lon_excess == expected_result)


def test_coordinates(location_generator):
    expected_result = [Coordinate(lat=2.9926330835665595, lon=1.8079467598322925), Coordinate(lat=2.9926330835665595, lon=13.0), Coordinate(lat=2.9926330835665595, lon=24.19205324016771), Coordinate(lat=10.5, lon=1.8079467598322925), Coordinate(lat=10.5, lon=13.0), Coordinate(lat=10.5, lon=24.19205324016771), Coordinate(lat=18.00736691643344, lon=1.8079467598322925), Coordinate(lat=18.00736691643344, lon=13.0), Coordinate(lat=18.00736691643344, lon=24.19205324016771)]

    assert (location_generator.coordinates == expected_result)
