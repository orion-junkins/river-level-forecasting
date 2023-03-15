from datetime import datetime, timedelta

from darts.timeseries import TimeSeries
import pandas as pd

from rlf.models.ensemble import Ensemble


class MockLinearRegressionModel:
    def __init__(self, expected_fit_arguments):
        self._expected_fit_arguments = expected_fit_arguments

    def fit(self, series, future_covariates):
        assert all((self._expected_fit_arguments["series"] == series.values()).flatten())
        assert all((self._expected_fit_arguments["future_covariates"] == future_covariates.values()).flatten())


class MockContributingModel:
    def __init__(
            self,
            expected_fit_arguments,
            input_chunk_length,
            expected_historical_forecasts_arguments) -> None:
        self._expected_fit_arguments = expected_fit_arguments
        self.fit_calls = 0

        self.input_chunk_length = input_chunk_length

        self._expected_historical_forecasts_arguments = expected_historical_forecasts_arguments

    def fit(self, series, future_covariates):
        assert all((self._expected_fit_arguments[self.fit_calls]["series"] == series.values()).flatten())
        assert all((self._expected_fit_arguments[self.fit_calls]["future_covariates"] == future_covariates.values()).flatten())
        self.fit_calls += 1

    def historical_forecasts(
            self,
            series,
            future_covariates,
            start,
            last_points_only,
            retrain,
            forecast_horizon,
            stride):
        assert last_points_only is True
        assert retrain is False

        assert all((self._expected_historical_forecasts_arguments["series"] == series.values()).flatten())
        assert all((self._expected_historical_forecasts_arguments["future_covariates"] == future_covariates.values()).flatten())
        assert self._expected_historical_forecasts_arguments["start"] == start
        assert self._expected_historical_forecasts_arguments["forecast_horizon"] == forecast_horizon
        assert self._expected_historical_forecasts_arguments["stride"] == stride

        return generate_hourly_time_series(datetime(2023, 1, 1, 1), [[9.0], [10.0]])


def generate_hourly_time_series(begin_date, data):
    total_len = len(data)
    one_hour_delta = timedelta(hours=1.0)
    dates = [begin_date + one_hour_delta * n for n in range(total_len)]

    data_dict = {"datetime": dates}
    for col in range(len(data[0])):
        data_dict[f"c{col}"] = []

    for row in range(total_len):
        for col in range(len(data[row])):
            data_dict[f"c{col}"].append(data[row][col])

    df = pd.DataFrame(data_dict)

    return TimeSeries.from_dataframe(df, time_col="datetime", freq="H")


def test_ensemble_fit_future_covariates():
    y = generate_hourly_time_series(datetime(2023, 1, 1), [[0.0], [1.0], [2.0]])
    X = generate_hourly_time_series(datetime(2023, 1, 1), [[3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])

    linear_regression_model = MockLinearRegressionModel(
        expected_fit_arguments={
            "series": [[1.0], [2.0]],
            "future_covariates": [[9.0], [10.0]]})

    contributing_models = [
        MockContributingModel(
            expected_fit_arguments=[
                {"series": [[0.0]], "future_covariates": [[3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]},
                {"series": [[0.0], [1.0], [2.0]], "future_covariates": [[3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]}
            ],
            input_chunk_length=1,
            expected_historical_forecasts_arguments={
                "series": [[0.0], [1.0], [2.0]],
                "future_covariates": [[3.0, 4.0], [5.0, 6.0], [7.0, 8.0]],
                "start": 2,
                "forecast_horizon": 1,
                "stride": 2}
        )
    ]

    ensemble = Ensemble(
        linear_regression_model,
        contributing_models,
        combiner_holdout_size=2,
        target_horizon=1,
        combiner_train_stride=2)

    ensemble.fit(y, future_covariates=X)

    assert contributing_models[0].fit_calls == 2
