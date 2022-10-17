import pytest

from forecasting.data_fetching_utilities.coordinate import Coordinate
from forecasting.data_fetching_utilities.location_generator import \
    LocationGenerator


@pytest.fixture
def bottom_left():
    return Coordinate(0, 0)


@pytest.fixture
def top_right():
    return Coordinate(21, 11)


def test_negative_separation_distance_is_invalid(bottom_left, top_right):
    separation_distance = -1
    with pytest.raises(ValueError):
        LocationGenerator(bottom_left, top_right, separation_distance_km=separation_distance)
