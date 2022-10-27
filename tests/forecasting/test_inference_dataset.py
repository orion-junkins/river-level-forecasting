import pytest

from rlf.forecasting.catchment_data import CatchmentData
from rlf.forecasting.inference_dataset import InferenceDataset
from rlf.forecasting.training_dataset import TrainingDataset
from fake_providers import FakeLevelProvider, FakeWeatherProvider


@pytest.fixture
def catchment_data():
    CatchmentData("test_catchment", FakeWeatherProvider(), FakeLevelProvider())


@pytest.fixture
def training_dataset(catchment_data):
    return TrainingDataset(catchment_data=catchment_data)


@pytest.fixture
def scalers(training_dataset):
    return (training_dataset.scaler, training_dataset.target_scaler)


@pytest.fixture
def inference_dataset(scalers):
    return InferenceDataset(scaler=scalers[0], target_scaler=scalers[1])


def test_data_scaled_0_1(inference_dataset):
    inference_dataset

    assert (inference_dataset.y.values().min() >= -0.1)
    assert (inference_dataset.y.values().max() <= 1.1)

    for x in inference_dataset.Xs:
        assert (x.values().min() >= -0.1)
        assert (x.values().max() <= 1.1)


def test_feature_engineering(catchment_data):
    column = "weather_attr_1"
    rolling_window_size = 3
    rolling_sum_column = column + "_sum_" + str(rolling_window_size)
    rolling_mean_column = column + "_mean_" + str(rolling_window_size)

    train_ds = InferenceDataset(catchment_data=catchment_data, rolling_sum_columns=[column], rolling_mean_columns=[column], rolling_window_sizes=[rolling_window_size])

    for x in train_ds.Xs:
        x_df = x.pd_dataframe()
        assert (x_df[column][-rolling_window_size:].sum() == x_df[rolling_sum_column][-1])

        assert (x_df[column][-rolling_window_size:].mean() == x_df[rolling_mean_column][-1])
