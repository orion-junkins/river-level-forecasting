from darts import TimeSeries
import numpy as np

from rlf.models.contributing_model import ContributingModel


class MockModel:
    def __init__(self, expected_columns: list[str]):
        self.expected_columns = expected_columns

    def fit(self, past_covariates, future_covariates, **kwargs) -> "MockModel":
        assert(sorted(past_covariates.columns) == sorted(self.expected_columns))
        assert(sorted(future_covariates.columns) == sorted(self.expected_columns))
        return self

    def predict(self, past_covariates, future_covariates, **kwargs) -> None:
        assert(sorted(past_covariates.columns) == sorted(self.expected_columns))
        assert(sorted(future_covariates.columns) == sorted(self.expected_columns))
        return None


def test_tributary_model_fit_separate_columns():
    expected_columns = ["1_a", "1_b"]
    all_columns = ["1_a", "1_b", "2_a", "2_b", "a"]
    values = np.array([[0, 1, 2, 3, 4],[5, 6, 7, 8, 9]])

    tm = ContributingModel(MockModel(expected_columns), "1_")
    covariates = TimeSeries.from_values(values, columns=all_columns)

    tm.fit(series=None, past_covariates=covariates, future_covariates=covariates)


def test_tributary_model_predict_separate_columns():
    expected_columns = ["1_a", "1_b"]
    all_columns = ["1_a", "1_b", "2_a", "2_b", "a"]
    values = np.array([[0, 1, 2, 3, 4],[5, 6, 7, 8, 9]])

    tm = ContributingModel(MockModel(expected_columns), "1_")
    covariates = TimeSeries.from_values(values, columns=all_columns)

    tm.predict(n=0, series=None, past_covariates=covariates, future_covariates=covariates)


def test_tributary_model_predict_do_not_separate_columns():
    expected_columns = ["1_a", "1_b", "2_a", "2_b", "a"]
    all_columns = ["1_a", "1_b", "2_a", "2_b", "a"]
    values = np.array([[0, 1, 2, 3, 4],[5, 6, 7, 8, 9]])

    tm = ContributingModel(MockModel(expected_columns))
    covariates = TimeSeries.from_values(values, columns=all_columns)

    tm.predict(n=0, series=None, past_covariates=covariates, future_covariates=covariates)


def test_tributary_model_fit_do_not_separate_columns():
    expected_columns = ["1_a", "1_b", "2_a", "2_b", "a"]
    all_columns = ["1_a", "1_b", "2_a", "2_b", "a"]
    values = np.array([[0, 1, 2, 3, 4],[5, 6, 7, 8, 9]])

    tm = ContributingModel(MockModel(expected_columns))
    covariates = TimeSeries.from_values(values, columns=all_columns)

    tm.fit(series=None, past_covariates=covariates, future_covariates=covariates)
