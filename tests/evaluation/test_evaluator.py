import numpy as np
import pandas as pd
from pandas import Timedelta, Timestamp
import pytest
from statistics import mean

from rlf.evaluating.evaluator import Evaluator


@pytest.fixture
def df():
    return (
        pd.DataFrame(
            {'level_true': {Timestamp('2023-01-01 01:00:00'): 10,
                            Timestamp('2023-01-01 02:00:00'): 20,
                            Timestamp('2023-01-01 03:00:00'): 30,
                            Timestamp('2023-01-01 04:00:00'): 40,
                            Timestamp('2023-01-01 05:00:00'): 50,
                            Timestamp('2023-01-01 06:00:00'): None},
             '23-01-01_02-00': {Timestamp('2023-01-01 01:00:00'): None,
                                Timestamp('2023-01-01 02:00:00'): 20,
                                Timestamp('2023-01-01 03:00:00'): 30,
                                Timestamp('2023-01-01 04:00:00'): 40,
                                Timestamp('2023-01-01 05:00:00'): None,
                                Timestamp('2023-01-01 06:00:00'): None},
             '23-01-01_03-00': {Timestamp('2023-01-01 01:00:00'): None,
                                Timestamp('2023-01-01 02:00:00'): None,
                                Timestamp('2023-01-01 03:00:00'): 33,
                                Timestamp('2023-01-01 04:00:00'): 42,
                                Timestamp('2023-01-01 05:00:00'): 52,
                                Timestamp('2023-01-01 06:00:00'): None},
             '23-01-01_04-00': {Timestamp('2023-01-01 01:00:00'): None,
                                Timestamp('2023-01-01 02:00:00'): None,
                                Timestamp('2023-01-01 03:00:00'): None,
                                Timestamp('2023-01-01 04:00:00'): 39,
                                Timestamp('2023-01-01 05:00:00'): 51,
                                Timestamp('2023-01-01 06:00:00'): 59}}
        )
    )


@pytest.fixture
def evaluator(df):
    eval = Evaluator(df)
    return eval


def test_level_true(evaluator):
    level_true = evaluator.level_true()
    assert level_true.shape == (4,)
    assert (level_true == [20, 30, 40, 50]).all()


def test_all_level_preds(evaluator):
    all_level_preds = evaluator.all_level_preds()
    assert all_level_preds.shape == (4, 3)
    assert (all_level_preds.columns == ["23-01-01_02-00", "23-01-01_03-00", "23-01-01_04-00"]).all()

    np.testing.assert_equal(all_level_preds["23-01-01_02-00"].values, np.array([20., 30., 40., np.nan]))
    np.testing.assert_equal(all_level_preds["23-01-01_03-00"].values, np.array([np.nan, 33., 42., 52.]))
    np.testing.assert_equal(all_level_preds["23-01-01_04-00"].values, np.array([np.nan, np.nan, 39., 51.]))


@pytest.fixture
def expected_absolute_errors_by_window():
    return {Timedelta('0 days 00:00:00'): [0.0, 3.0, 1.0],
            Timedelta('0 days 01:00:00'): [0.0, 2.0, 1.0],
            Timedelta('0 days 02:00:00'): [0.0, 2.0]}


def test_absolute_errors_by_window(evaluator, expected_absolute_errors_by_window):
    assert (evaluator.absolute_errors_by_window() == expected_absolute_errors_by_window)


@pytest.fixture
def expected_percent_errors_by_window():
    return {Timedelta('0 days 00:00:00'): [0.0, 0.1, 0.025],
            Timedelta('0 days 01:00:00'): [0.0, 0.05, 0.02],
            Timedelta('0 days 02:00:00'): [0.0, 0.04]}


def test_percent_errors_by_window(evaluator, expected_percent_errors_by_window):
    assert (evaluator.percent_errors_by_window() == expected_percent_errors_by_window)


@pytest.fixture
def expected_mae_by_window():
    return {Timedelta('0 days 00:00:00'): mean([0.0, 3.0, 1.0]),
            Timedelta('0 days 01:00:00'): mean([0.0, 2.0, 1.0]),
            Timedelta('0 days 02:00:00'): mean([0.0, 2.0])}


def test_mae_by_window(evaluator, expected_mae_by_window):
    assert (evaluator.mae_by_window() == expected_mae_by_window)


@pytest.fixture
def expected_mape_by_window():
    return {Timedelta('0 days 00:00:00'): mean([0.0, 0.1, 0.025]),
            Timedelta('0 days 01:00:00'): mean([0.0, 0.05, 0.02]),
            Timedelta('0 days 02:00:00'): mean([0.0, 0.04])}


def test_mape_by_window(evaluator, expected_mape_by_window):
    assert (evaluator.mape_by_window() == expected_mape_by_window)


def test_df_mae(evaluator, expected_mae_by_window):
    expected = pd.DataFrame.from_dict(expected_mae_by_window(), orient='index').sort_index()

    assert (evaluator.df_mae().equals(expected))


def test_df_mape(evaluator, expected_mape_by_window):
    expected = pd.DataFrame.from_dict(expected_mape_by_window, orient='index').sort_index()

    assert (evaluator.df_mape().equals(expected))


@pytest.fixture
def df_with_zeros():
    return (
        pd.DataFrame(
            {'level_true': {Timestamp('2023-01-01 01:00:00'): 0,
                            Timestamp('2023-01-01 02:00:00'): 20,
                            Timestamp('2023-01-01 03:00:00'): 0,
                            Timestamp('2023-01-01 04:00:00'): 40,
                            Timestamp('2023-01-01 05:00:00'): 50,
                            Timestamp('2023-01-01 06:00:00'): None},
             '23-01-01_02-00': {Timestamp('2023-01-01 01:00:00'): None,
                                Timestamp('2023-01-01 02:00:00'): 20,
                                Timestamp('2023-01-01 03:00:00'): 30,
                                Timestamp('2023-01-01 04:00:00'): 40,
                                Timestamp('2023-01-01 05:00:00'): None,
                                Timestamp('2023-01-01 06:00:00'): None}}))


def test_mape_with_zeros(df_with_zeros):
    eval = Evaluator(df_with_zeros)
    with pytest.raises(ZeroDivisionError):
        eval.errors_grouped_by_window(absolute=False)
