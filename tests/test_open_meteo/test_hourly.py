import pytest

from forecasting.data_fetching_utilities.open_meteo.hourly import Hourly


class TestHourly():

    @pytest.fixture
    def hourly_parameters(self):
        return Hourly()

    def test_has_variables_attribute(self, hourly_parameters):
        assert hasattr(hourly_parameters, "variables")

    def test_variables_attribute_is_list(self, hourly_parameters):
        assert isinstance(
            hourly_parameters.variables, list)

    def test_variables_attribute_is_empty_list(self, hourly_parameters):
        assert hourly_parameters.variables == []
