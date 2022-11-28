from datetime import datetime

import pandas as pd

from rlf.forecasting.base_dataset import BaseDataset

class FakeCatchmentData:
    def __init__(self):
        self.name = "test_name"


def test_base_dataset_add_engineered_features():
    bf = BaseDataset(FakeCatchmentData())
    data = {
        "c": [1.0]
    }

    df = pd.DataFrame(data, index=[datetime(2022, 1, 1, 1)])

    df = bf._add_engineered_features(df)

    assert "day_of_year" in df.columns
    assert df["day_of_year"][0] == 1


def test_base_dataset_add_engineered_features_drop_na():
    bf = BaseDataset(FakeCatchmentData())
    data = {
        "c": [1.0, None, float("nan")]
    }

    df = pd.DataFrame(data, index=[datetime(2022, 1, 1, 1), datetime(2022, 1, 1, 2), datetime(2022, 1, 1, 3)])

    df = bf._add_engineered_features(df)

    assert len(df) == 1
    assert df.index[0] == datetime(2022, 1, 1, 1)
