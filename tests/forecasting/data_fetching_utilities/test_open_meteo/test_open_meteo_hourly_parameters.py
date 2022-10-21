import pytest

from rlf.forecasting.data_fetching_utilities.open_meteo.open_meteo_hourly_parameters import OpenMeteoHourlyParameters


class TestHourly():

    @pytest.fixture
    def hourly_parameters(self):
        return OpenMeteoHourlyParameters()

    def test_has_variables_attribute(self, hourly_parameters):
        assert hasattr(hourly_parameters, "variables")

    def test_variables_attribute_is_list(self, hourly_parameters):
        assert isinstance(
            hourly_parameters.variables, list)

    def test_variables_attribute_is_empty_list(self, hourly_parameters):
        assert hourly_parameters.variables == []
