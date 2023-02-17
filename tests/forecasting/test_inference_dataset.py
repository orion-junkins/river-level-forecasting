import pytest

from rlf.forecasting.catchment_data import CatchmentData
from rlf.forecasting.inference_dataset import InferenceDataset
from rlf.forecasting.training_dataset import TrainingDataset
from fake_providers import FakeLevelProvider, FakeWeatherProvider


@pytest.fixture
def catchment_data():
    return CatchmentData("test_catchment", FakeWeatherProvider(num_historical_samples=1000), FakeLevelProvider(num_historical_samples=1000))


@pytest.fixture
def training_dataset(catchment_data):
    return TrainingDataset(catchment_data=catchment_data)


@pytest.fixture
def scalers(training_dataset):
    return (training_dataset.scaler, training_dataset.target_scaler)


@pytest.fixture
def inference_dataset(scalers, catchment_data):
    return InferenceDataset(catchment_data=catchment_data, scaler=scalers[0], target_scaler=scalers[1])


def test_data_scaled_0_1(inference_dataset):
    inference_dataset

    assert (inference_dataset.y.values().min() >= -0.1)
    assert (inference_dataset.y.values().max() <= 1.1)

    assert inference_dataset.X.values().min() >= -0.1
    assert inference_dataset.X.values().max() <= 1.1
