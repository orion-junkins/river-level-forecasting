import pytest

from forecasting.data_fetching_utilities.open_meteo.models import ResponseModel, HourlyParametersModel, HourlyUnitsModel


class TestResponseModel():

    @pytest.fixture
    def response(self):
        return ResponseModel()

    def test_initialization(self, response):
        try:
            response
        except NameError:
            pytest.fail("ResponseModel failed to initialize")

    def test_has_type_hourly_units_or_empty(self, response):
        assert isinstance(response.hourly_units, list)
        response.hourly_units = HourlyUnitsModel()
        assert isinstance(response.hourly_units, HourlyUnitsModel)

    def test_has_type_hourly_or_empty(self, response):
        assert isinstance(response.hourly_parameters, list)
        response.hourly_parameters = HourlyParametersModel()
        assert isinstance(response.hourly_parameters, HourlyParametersModel)


class TestHourlyParametersModel():

    @pytest.fixture
    def hourly(self):
        return HourlyParametersModel()

    def test_initialization(self, hourly):
        try:
            hourly
        except NameError:
            pytest.fail("HourlyParametersModel failed to initialize")


class TestHourlyUnitsModel():

    @pytest.fixture
    def hourly_units(self):
        return HourlyUnitsModel()

    def test_initialization(self, hourly_units):
        try:
            hourly_units
        except NameError:
            pytest.fail("HourlyUnitsModel failed to initialize")
