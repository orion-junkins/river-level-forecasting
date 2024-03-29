import pytest

from rlf.forecasting.catchment_data import CatchmentData
from rlf.forecasting.inference_dataset import InferenceDataset
from rlf.forecasting.training_dataset import TrainingDataset
from rlf.forecasting.inference_forecaster import InferenceForecaster
from fake_providers import FakeLevelProvider, FakeWeatherProvider


class FakeObject:
    def __init__(self, attributes: dict) -> None:
        self._fake_attributes = attributes

    def __getattr__(self, __name: str):
        return self._fake_attributes[__name]


class MockModel:
    def __init__(self, expected_n, expected_series, expected_future_covariates, prediction):
        self.expected_n = expected_n
        self.expected_series = expected_series
        self.expected_future_covariates = expected_future_covariates
        self.prediction = prediction
        self.contributing_models = [FakeObject({
            "_column_prefix": "prefix_",
            "_base_model": FakeObject({
                "future_covariate_series": FakeObject({
                    "columns": ["prefix_A", "prefix_B"]
                })
            })
        })]

    def predict(self, n, series, future_covariates=None, past_covariates=None):
        print(series)
        print(self.expected_series)
        assert n == self.expected_n
        assert series == self.expected_series
        assert future_covariates == self.expected_future_covariates
        assert past_covariates is None
        return self.prediction


class FakeScaler:
    def __init__(self, scaler):
        self._scaler = scaler

    def transform(self, X):
        if isinstance(X, list):
            return X
        return self._scaler.transform(X)

    def inverse_transform(self, X):
        if isinstance(X, list):
            return X
        return self._scaler.inverse_transform(X)


class FakeInferenceForecaster(InferenceForecaster):

    def __init__(self, mock_model, scalers, *args, **kwargs):
        self._mock_model = mock_model
        self._scalers = scalers
        super().__init__(*args, **kwargs)

    def _load_ensemble(self, load_cpu):
        return self._mock_model

    def _load_scalers(self):
        return {"scaler": self._scalers[0], "target_scaler": self._scalers[1]}

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
    return TrainingDataset(catchment_data=catchment_data, validation_size=10, test_size=10)


@pytest.fixture
def scalers(training_dataset):
    return (training_dataset.scaler, training_dataset.target_scaler)


@pytest.fixture
def inference_dataset(scalers, catchment_data):
    return InferenceDataset(catchment_data=catchment_data, scaler=scalers[0], target_scaler=scalers[1])


def test_inference_forecaster_init(catchment_data, scalers):
    inference_forecaster = FakeInferenceForecaster(None, scalers, catchment_data=catchment_data)

    assert (type(inference_forecaster.dataset) is InferenceDataset)


def test_inference_forecaster_predict(inference_dataset, catchment_data, scalers):
    scalers = [FakeScaler(scalers[0]), FakeScaler(scalers[1])]
    mock_model = MockModel(24, inference_dataset.y, inference_dataset.X, [n for n in range(24)])
    inference_forecaster = FakeInferenceForecaster(mock_model, scalers, catchment_data=catchment_data)
    actual_results = inference_forecaster.predict()
    expected_results = [n for n in range(24)]

    assert actual_results == expected_results
