import pytest
from forecasting.data_fetching_utilities.api.exceptions import RestInvokerException


class TestExceptions:

    def test_rest_invoker_exception(self):
        with pytest.raises(RestInvokerException):
            raise RestInvokerException
