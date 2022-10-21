import pytest
import pandas as pd

from rlf.forecasting.data_fetching_utilities.data_providers.datum import Datum


class TestDatum():

    @pytest.fixture
    def datum(self):
        return Datum(longitude=0, latitude=0, elevation=0, utc_offset_seconds=0,
                     timezone="UTC", hourly_parameters={})

    def test_initialization(self, datum):
        try:
            datum
        except Exception as e:
            pytest.fail("Datum() raised an exception: " + str(e))

    def test_get_hourly_parameters_returns_dict(self, datum):
        assert isinstance(datum.get_hourly_parameters(
            as_pandas_data_frame=False), dict)

    def test_get_hourly_parameters_returns_dataframe(self, datum):
        assert isinstance(datum.get_hourly_parameters(
            as_pandas_data_frame=True), pd.DataFrame)

    def test_get_hourly_parameters_returns_dataframe_with_correct_columns(self, datum):
        assert datum.get_hourly_parameters(
            as_pandas_data_frame=True).columns.tolist() == list(datum.hourly_parameters.keys())

    def test_has_hourly_parameters(self, datum):
        assert datum.hourly_parameters == {}

    def test_returns_longitude(self, datum):
        assert datum.get_longitude() == 0

    def test_returns_latitude(self, datum):
        assert datum.get_latitude() == 0

    def test_returns_elevation(self, datum):
        assert datum.get_elevation() == 0

    def test_returns_utc_offset_seconds(self, datum):
        assert datum.get_utc_offset_seconds() == 0

    def test_returns_timezone(self, datum):
        assert datum.get_timezone() == "UTC"
