from datetime import datetime
import pytest
import pytz

from rlf.forecasting.data_fetching_utilities.level_provider.level_provider_nwis import LevelProviderNWIS


@pytest.fixture
def level_provider():
    return LevelProviderNWIS("14377100")


def test_fetch_recent_level(level_provider):
    expected_num_samples = 5
    df = level_provider.fetch_recent_level(expected_num_samples)

    # One row per requested hour
    assert (df.shape[0] == expected_num_samples)

    # Only one column
    assert (df.shape[1] == 1)

    # Hourly frequency data
    assert (df.index.freq == 'H')

    # UTC timezone
    assert (str(df.index[0].tz) == 'UTC')


def test_fetch_recent_level_from_reference(level_provider):
    reference_timestamp = '22-12-01_12-00'
    level_provider.set_timestamp(reference_timestamp)
    reference_dt = datetime.strptime(reference_timestamp, '%y-%m-%d_%H-%M').replace(tzinfo=pytz.utc)

    expected_num_samples = 5
    df = level_provider.fetch_recent_level(expected_num_samples)

    # One row per requested hour
    assert (df.shape[0] == expected_num_samples)

    # Only one column
    assert (df.shape[1] == 1)

    # Hourly frequency data
    assert (df.index.freq == 'H')

    # UTC timezone
    assert (str(df.index[0].tz) == 'UTC')

    # No data beyond reference timestamp
    for index_value in df.index:
        assert (index_value <= reference_dt)
