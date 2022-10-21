import pytest
from rlf.forecasting.data_fetching_utilities.weather_provider.api.exceptions import RestInvokerException
from rlf.forecasting.data_fetching_utilities.weather_provider.api.models import Response
from rlf.forecasting.data_fetching_utilities.weather_provider.api.rest_invoker import RestInvoker


class TestRestInvoker():

    @pytest.fixture
    def rest_adapter(self):
        return RestInvoker(protocol="fake protocol", hostname="fake hostname", version="fake version", ssl_verify=True)

    def test_get_fails(self, rest_adapter):
        with pytest.raises(RestInvokerException):
            rest_adapter.get("fake url")

    def test_get_fails_with_message(self, rest_adapter):
        with pytest.raises(RestInvokerException):
            rest_adapter.get("fake url")

    def test_get_returns_type_response(self):
        invoker = RestInvoker(
            protocol="https", hostname="jsonplaceholder.typicode.com", version=None, ssl_verify=True)
        assert isinstance(invoker.get(path="posts"), Response)

    def test_get_returns_response_with_status_code(self):
        invoker = RestInvoker(
            protocol="https", hostname="jsonplaceholder.typicode.com", version=None, ssl_verify=True)
        assert invoker.get(path="posts").status_code == 200

    def test_get(self):
        invoker = RestInvoker(
            protocol="https", hostname="jsonplaceholder.typicode.com", version=None, ssl_verify=True)
        res = invoker.get(path="posts")
        assert res.url == "https://jsonplaceholder.typicode.com/posts"
