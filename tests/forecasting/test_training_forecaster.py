import json
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
    assert os.path.exists(os.path.join(tmp_path, "test_catchment", "metadata"))


def test_training_forecaster_save_model_scaler_is_correct(tmp_path, training_dataset):
    training_forecaster = TrainingForecaster(
        RegressionEnsembleModel([LinearRegressionModel(lags=1)], 10),
        training_dataset,
        root_dir=tmp_path
    )

    training_forecaster.save_model()
    with open(os.path.join(tmp_path, "test_catchment", "scaler"), "rb") as f:
        scaler = pickle.load(f)

    assert isinstance(scaler, dict)
    assert "scaler" in scaler
    assert isinstance(scaler["scaler"], Scaler)
    assert "target_scaler" in scaler
    assert isinstance(scaler["target_scaler"], Scaler)


def test_training_forecaster_save_model_metadata_is_correct(tmp_path, training_dataset):
    training_forecaster = TrainingForecaster(
        RegressionEnsembleModel([LinearRegressionModel(lags=1)], 10),
        training_dataset,
        root_dir=tmp_path
    )

    training_forecaster.save_model()
    with open(os.path.join(tmp_path, "test_catchment", "metadata")) as f:
        metadata = json.load(f)

    assert "api_columns" in metadata
    assert metadata["api_columns"] == ['weather_attr_1', 'weather_attr_2']
    assert "engineered_columns" in metadata
    assert metadata["engineered_columns"] == ['day_of_year']
    assert "mean_columns" in metadata
    assert metadata["mean_columns"] == []
    assert "sum_columns" in metadata
    assert metadata["sum_columns"] == []
    assert "windows" in metadata
    assert metadata["windows"] == [240, 720]


def test_training_forecaster_save_model_metadata_with_rolling_columns_correct(tmp_path, training_dataset):
    training_forecaster = TrainingForecaster(
        RegressionEnsembleModel([LinearRegressionModel(lags=1)], 10),
        training_dataset,
        root_dir=tmp_path
    )

    training_forecaster.dataset.rolling_mean_columns = ["mean_col_1", "mean_col_2"]
    training_forecaster.dataset.rolling_sum_columns = ["sum_col_1", "sum_col_2"]

    training_forecaster.save_model()
    with open(os.path.join(tmp_path, "test_catchment", "metadata")) as f:
        metadata = json.load(f)

    assert metadata["mean_columns"] == ["mean_col_1", "mean_col_2"]
    assert metadata["sum_columns"] == ["sum_col_1", "sum_col_2"]
