import pytest

from rlf.forecasting.catchment_data import CatchmentData
from rlf.forecasting.training_dataset import TrainingDataset
from fake_providers import FakeLevelProvider, FakeWeatherProvider


@pytest.fixture
def catchment_data():
    return CatchmentData("test_catchment", FakeWeatherProvider(), FakeLevelProvider())


def test_partition_size():
    num_historical_samples = 10
    weather_provider = FakeWeatherProvider(num_historical_samples)
    level_provider = FakeLevelProvider(num_historical_samples)
    catchment_data = CatchmentData("test_catchment", weather_provider, level_provider)

    test_size = 1
    validation_size = 2
    train_ds = TrainingDataset(catchment_data=catchment_data, test_size=test_size, validation_size=validation_size)

    assert (len(train_ds.y_test) == test_size)
    assert (len(train_ds.y_validation) == validation_size)
    assert (len(train_ds.y_train) == num_historical_samples - test_size - validation_size)


def test_excess_level_data_dropped():
    num_weather_samples = 4
    weather_provider = FakeWeatherProvider(num_historical_samples=num_weather_samples)

    num_level_samples = 8
    level_provider = FakeLevelProvider(num_historical_samples=num_level_samples)

    catchment_data = CatchmentData("test_catchment", weather_provider, level_provider)
    train_ds = TrainingDataset(catchment_data=catchment_data, test_size=1, validation_size=2)
    print(train_ds.y)
    print(train_ds.X)
    assert (len(train_ds.y) == min(num_weather_samples, num_level_samples))
    assert (len(train_ds.X) == min(num_weather_samples, num_level_samples))


def test_excess_weather_data_dropped():
    num_weather_samples = 8
    weather_provider = FakeWeatherProvider(num_historical_samples=num_weather_samples)

    num_level_samples = 4
    level_provider = FakeLevelProvider(num_historical_samples=num_level_samples)

    catchment_data = CatchmentData("test_catchment", weather_provider, level_provider)
    train_ds = TrainingDataset(catchment_data=catchment_data, test_size=1, validation_size=2)
    print(train_ds.y)
    print(train_ds.X)
    assert (len(train_ds.y) == min(num_weather_samples, num_level_samples))
    assert (len(train_ds.X) == min(num_weather_samples, num_level_samples))


def test_data_scaled_0_1(catchment_data):
    train_ds = TrainingDataset(catchment_data=catchment_data, test_size=1, validation_size=2)

    assert (pytest.approx(train_ds.y_train.values().min()) == 0.0)
    assert (pytest.approx(train_ds.y_train.values().max()) == 1.0)
    assert (train_ds.y_test.values().min() >= -0.1)
    assert (train_ds.y_test.values().max() <= 1.2)
    assert (train_ds.y_validation.values().min() >= -0.1)
    assert (train_ds.y_validation.values().max() <= 1.2)

    assert (pytest.approx(train_ds.X_train.values().min()) == 0.0)
    assert (pytest.approx(train_ds.X_train.values().max()) == 1.0)
    assert (train_ds.X_test.values().min() >= -0.1)
    assert (train_ds.X_test.values().max() <= 1.2)
    assert (train_ds.X_validation.values().min() >= -0.1)
    assert (train_ds.X_validation.values().max() <= 1.2)


def test_feature_engineering(catchment_data):
    train_ds = TrainingDataset(
        catchment_data=catchment_data,
        test_size=1,
        validation_size=2,
        rolling_sum_columns=["weather_attr_1"],
        rolling_mean_columns=["weather_attr_1"],
        rolling_window_sizes=[3]
    )

    x_df = train_ds.X.pd_dataframe()

    assert (x_df["0.10_1.10_weather_attr_1_sum_3"][-1] == 84.0)
    assert (x_df["0.10_1.10_weather_attr_1_mean_3"][-1] == 28.0)


def test_correct_precision(catchment_data):
    train_ds = TrainingDataset(catchment_data=catchment_data, test_size=1, validation_size=2)

    assert train_ds.X.dtype == "float32"
    assert train_ds.y.dtype == "float32"

    assert train_ds.X_train.dtype == "float32"
    assert train_ds.X_test.dtype == "float32"
    assert train_ds.X_validation.dtype == "float32"
    assert train_ds.y_train.dtype == "float32"
    assert train_ds.y_test.dtype == "float32"
    assert train_ds.y_validation.dtype == "float32"


def test_invalid_dataset_size_raises_error():
    num_historical_samples = 10
    weather_provider = FakeWeatherProvider(num_historical_samples)
    level_provider = FakeLevelProvider(num_historical_samples)
    catchment_data = CatchmentData("test_catchment", weather_provider, level_provider)

    test_size = 6
    validation_size = 6
    with pytest.raises(ValueError):
        TrainingDataset(catchment_data=catchment_data, test_size=test_size, validation_size=validation_size)
