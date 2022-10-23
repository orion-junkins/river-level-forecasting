import pytest

from rlf.forecasting.data_fetching_utilities.level_provider.LevelProviderNWIS import LevelProviderNWIS


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
