import pytest

from rlf.forecasting.data_fetching_utilities.weather_provider.api.exceptions import RestInvokerException


def test_rest_invoker_exception():
    with pytest.raises(RestInvokerException):
        raise RestInvokerException
