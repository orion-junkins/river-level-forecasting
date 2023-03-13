import pytest

from rlf.models.ensemble import Ensemble


class MockLinearRegressionModel:
    def __init__(self, expected_fit_arguments):
        self._expected_fit_arguments = expected_fit_arguments

    def fit(self, series, future_covariates):
        assert self._expected_fit_arguments["series"] == series
        assert self._expected_fit_arguments["future_covariates"] == future_covariates


class MockContributingModel:
    def __init__(
            self,
            expected_fit_arguments,
            input_chunk_length,
            expected_historical_forecasts_arguments
        ) -> None:
        self._expected_fit_arguments = expected_fit_arguments
        self.fit_calls = 0

        self.input_chunk_length = input_chunk_length

        self._expected_historical_forecasts_arguments = expected_historical_forecasts_arguments

    def fit(self, series, future_covariates):
        assert self._expected_fit_arguments[self.fit_calls]["series"] == series
        assert self._expected_fit_arguments[self.fit_calls]["future_covariates"] == future_covariates
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

        assert self._expected_historical_forecasts_arguments["series"] == series
        assert self._expected_historical_forecasts_arguments["future_covariates"] == future_covariates
        assert self._expected_historical_forecasts_arguments["start"] == start
        assert self._expected_historical_forecasts_arguments["forecast_horizon"] == forecast_horizon
        assert self._expected_historical_forecasts_arguments["stride"] == stride

        return ["a", "b"]


def test_ensemble_fit_future_covariates():
    y = [0.0, 1.0]
    X = [[2.0, 3.0], [4.0, 5.0]]

    linear_regression_model = MockLinearRegressionModel(expected_fit_arguments={"series": ["a", "b"], "future_covariates": ["stacked forecasts"]})
    contributing_models = [
        MockContributingModel(
            expected_fit_arguments = [
                {"series": [0.0], "future_covariates": [[2.0, 3.0], [4.0, 5.0]]},
                {"series": [0.0, 1.0], "future_covariates": [[2.0, 3.0], [4.0, 5.0]]}
            ],
            input_chunk_length=1,
            expected_historical_forecasts_arguments = {
                "series": [0.0, 1.0],
                "future_covariates": [[2.0, 3.0], [4.0, 5.0]],
                "start": 2,
                "forecast_horizon": 1,
                "stride": 2}
        )
    ]

    ensemble = Ensemble(
        linear_regression_model,
        contributing_models,
        combiner_holdout_size=1,
        target_horizon=1,
        combiner_train_stride=2)

    ensemble.fit(y, future_covariates=X)

    assert contributing_models[0].fit_calls == 2
