import pytest

from rlf.forecasting.catchment_data import CatchmentData
from rlf.forecasting.inference_dataset import InferenceDataset
from rlf.forecasting.training_dataset import TrainingDataset
from rlf.forecasting.inference_forecaster import InferenceForecaster
from fake_providers import FakeLevelProvider, FakeWeatherProvider


class FakeScaler:
    def __init__(self, scale):
        self._scale = scale

    def transform(self, X):
        return X * self._scale

class FakeInferenceForecaster(InferenceForecaster):

    def __init__(self, mock_model, *args, **kwargs):
        self._mock_model = mock_model
        super().__init__(*args, **kwargs)

    def _load_ensemble(self):
        return self._mock_model

    def _load_scalers(self):
        return {"scaler": FakeScaler(1.0), "target_scaler": FakeScaler(1.0)}

    def _load_metadata(self) -> dict:
        return {
            "api_columns": ["weather_attr_1", "weather_attr_2"],
            "engineered_columns": ["day_of_year"],
            "sum_columns": ['weather_attr_1'],
            "mean_columns": ['weather_attr_2'],
            "windows": []
        }


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


def test_inference_forecaster_init(catchment_data):
    inference_forecaster = FakeInferenceForecaster(None, catchment_data=catchment_data)

    assert (type(inference_forecaster.dataset) is InferenceDataset)
