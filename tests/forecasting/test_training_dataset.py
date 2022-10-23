import pytest

from rlf.forecasting.catchment_data import CatchmentData
from rlf.forecasting.training_dataset import TrainingDataset
from fake_providers import FakeLevelProvider, FakeWeatherProvider
from tests.forecasting.test_catchment_data import fake_weather_provider


@pytest.fixture
def catchment():
    CatchmentData("test_catchment", FakeWeatherProvider(), FakeLevelProvider())


def test_partition_size(catchment):
    num_historical_samples = 100
    weather_provider = FakeWeatherProvider(num_historical_samples)
    level_provider = FakeLevelProvider(num_historical_samples)
    catchment = CatchmentData("test_catchment", weather_provider, level_provider)
    train_ds = TrainingDataset(catchment)

    test_size = 0.1
    expected_num_test_elements = num_historical_samples * test_size

    assert (len(train_ds.y_test) == expected_num_test_elements)

