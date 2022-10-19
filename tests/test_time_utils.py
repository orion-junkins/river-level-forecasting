import pytest

<<<<<<< HEAD
from forecasting.general_utilities.time_utils import \
    convert_timestamp_to_datetime
=======
from rlf.forecasting.general_utilities.time_utils import convert_timestamp_to_datetime
>>>>>>> 3b51ef4 (new directory structure and pyproject.toml)


@pytest.fixture
def good_timestamp_int():
    return 1641038400


@pytest.fixture
def good_timestamp_str(good_timestamp_int):
    return str(good_timestamp_int)


def test_convert_timestamp_to_datetime_int(good_timestamp_int):
    expected_result = "2022-01-01 12:00:00"
    actual_result = convert_timestamp_to_datetime(good_timestamp_int)
    assert (actual_result == expected_result)


def test_convert_timestamp_to_datetime_str(good_timestamp_str):
    expected_result = "2022-01-01 12:00:00"
    actual_result = convert_timestamp_to_datetime(good_timestamp_str)
    assert (actual_result == expected_result)
