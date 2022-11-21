import pytest

from fake_providers import FakeLevelProvider, FakeWeatherProvider
from rlf.forecasting.catchment_data import CatchmentData
from rlf.forecasting.training_dataset import TrainingDataset
from rlf.forecasting.training_forecaster import TrainingForecaster


@pytest.fixture
def catchment_data():
    return CatchmentData("test_catchment", FakeWeatherProvider(num_historical_samples=1000), FakeLevelProvider(num_historical_samples=1000))


@pytest.fixture
def training_dataset(catchment_data):
    return TrainingDataset(catchment_data=catchment_data)


def test_training_forecaster_init(training_dataset, catchment_data):
    training_forecaster = TrainingForecaster(model=None, catchment_data=catchment_data, dataset=training_dataset)

    assert (type(training_forecaster.dataset) is TrainingDataset)
