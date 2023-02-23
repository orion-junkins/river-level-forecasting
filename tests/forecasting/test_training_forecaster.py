import os
import pickle

from darts.dataprocessing.transformers.scaler import Scaler
from darts.models.forecasting.linear_regression_model import LinearRegressionModel
from darts.models.forecasting.regression_ensemble_model import RegressionEnsembleModel
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
    return TrainingDataset(catchment_data)

def test_training_forecaster_init(training_dataset):
    training_forecaster = TrainingForecaster(model=None, dataset=training_dataset)

    assert (type(training_forecaster.dataset) is TrainingDataset)


def test_training_forecaster_save_model(tmp_path, training_dataset):
    training_forecaster = TrainingForecaster(
        RegressionEnsembleModel([LinearRegressionModel(lags=1)], 10),
        training_dataset,
        root_dir=tmp_path
    )

    training_forecaster.save_model()
    assert os.path.exists(os.path.join(tmp_path, "test_catchment"))
    assert os.path.exists(os.path.join(tmp_path, "test_catchment", "frcstr"))
    assert os.path.exists(os.path.join(tmp_path, "test_catchment", "scaler"))

    with open(os.path.join(tmp_path, "test_catchment", "scaler"), "rb") as f:
        scaler = pickle.load(f)

    assert isinstance(scaler, dict)
    assert "scaler" in scaler
    assert isinstance(scaler["scaler"], Scaler)
    assert "target_scaler" in scaler
    assert isinstance(scaler["target_scaler"], Scaler)
