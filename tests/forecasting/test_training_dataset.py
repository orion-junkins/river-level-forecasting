import pytest

from rlf.forecasting.catchment_data import CatchmentData
from rlf.forecasting.training_dataset import TrainingDataset
from fake_providers import FakeLevelProvider, FakeWeatherProvider


@pytest.fixture
def catchment_data():
    CatchmentData("test_catchment", FakeWeatherProvider(), FakeLevelProvider())


def test_partition_size():
    num_historical_samples = 10
    weather_provider = FakeWeatherProvider(num_historical_samples)
    level_provider = FakeLevelProvider(num_historical_samples)
    catchment_data = CatchmentData("test_catchment", weather_provider, level_provider)

    test_size = 0.1
    train_ds = TrainingDataset(catchment_data=catchment_data, test_size=test_size)
    expected_num_test_elements = num_historical_samples * test_size

    assert (len(train_ds.y_test) == expected_num_test_elements)


def test_excess_level_data_dropped():
    num_weather_samples = 4
    weather_provider = FakeWeatherProvider(num_historical_samples=num_weather_samples)

    num_level_samples = 8
    level_provider = FakeLevelProvider(num_historical_samples=num_level_samples)

    catchment_data = CatchmentData("test_catchment", weather_provider, level_provider)
    train_ds = TrainingDataset(catchment_data=catchment_data)
    print(train_ds.y)
    print(train_ds.Xs[0])
    assert (len(train_ds.y) == min(num_weather_samples, num_level_samples))
    for x in train_ds.Xs:
        assert (len(x) == min(num_weather_samples, num_level_samples))


def test_excess_weather_data_dropped():
    num_weather_samples = 8
    weather_provider = FakeWeatherProvider(num_historical_samples=num_weather_samples)

    num_level_samples = 4
    level_provider = FakeLevelProvider(num_historical_samples=num_level_samples)

    catchment_data = CatchmentData("test_catchment", weather_provider, level_provider)
    train_ds = TrainingDataset(catchment_data=catchment_data)
    print(train_ds.y)
    print(train_ds.Xs[0])
    assert (len(train_ds.y) == min(num_weather_samples, num_level_samples))
    for x in train_ds.Xs:
        assert (len(x) == min(num_weather_samples, num_level_samples))


def test_data_scaled_0_1(catchment_data):
    train_ds = TrainingDataset(catchment_data=catchment_data)

    assert (train_ds.y_train.values().min() >= 0.0)
    assert (train_ds.y_train.values().max() <= 1.0)
    assert (train_ds.y_test.values().min() >= -0.1)
    assert (train_ds.y_test.values().max() <= 1.1)

    for x_train, x_test in zip(train_ds.Xs_train, train_ds.Xs_test):
        assert (x_train.values().min() >= 0.0)
        assert (x_train.values().max() <= 1.0)
        assert (x_test.values().min() >= -0.1)
        assert (x_test.values().max() <= 1.1)


def test_feature_engineering(catchment_data):
    column = "weather_attr_1"
    rolling_window_size = 3
    rolling_sum_column = column + "_sum_" + str(rolling_window_size)
    rolling_mean_column = column + "_mean_" + str(rolling_window_size)

    train_ds = TrainingDataset(catchment_data=catchment_data, rolling_sum_columns=[column], rolling_mean_columns=[column], rolling_window_sizes=[rolling_window_size])

    for x in train_ds.Xs:
        x_df = x.pd_dataframe()
        assert (x_df[column][-rolling_window_size:].sum() == x_df[rolling_sum_column][-1])

        assert (x_df[column][-rolling_window_size:].mean() == x_df[rolling_mean_column][-1])
