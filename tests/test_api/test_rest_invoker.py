import pytest
from forecasting.data_fetching_utilities.api.exceptions import RestInvokerException
from forecasting.data_fetching_utilities.api.models import Response
from forecasting.data_fetching_utilities.api.rest_invoker import RestInvoker


class TestRestInvoker():

    @pytest.fixture
    def rest_adapter(self):
        return RestInvoker(protocol="fake protocol", hostname="fake hostname", version="fake version", ssl_verify=True)

    def test_initialization(self):
        try:
            RestInvoker()
        except NameError:
            pytest.fail("RestInvoker failed to initialize")

    def test_has_protocol(self, rest_adapter):
        assert hasattr(rest_adapter, "_protocol")

    def test_protocol_is_type_string(self, rest_adapter):
        assert isinstance(rest_adapter._protocol, str)

    def test_has_hostname(self, rest_adapter):
        assert hasattr(rest_adapter, "_hostname")

    def test_hostname_is_type_string(self, rest_adapter):
        assert isinstance(rest_adapter._hostname, str)

    def test_has_version(self, rest_adapter):
        assert hasattr(rest_adapter, "_version")

    def test_version_is_type_string(self, rest_adapter):
        assert isinstance(rest_adapter._version, str)

    def test_has_ssl_verify(self, rest_adapter):
        assert hasattr(rest_adapter, "_ssl_verify")

    def test_ssl_verify(self, rest_adapter):
        assert rest_adapter._ssl_verify is True

    def test_get_fails(self, rest_adapter):
        with pytest.raises(RestInvokerException):
            rest_adapter.get("fake url")

    def test_get_fails_with_message(self, rest_adapter):
        with pytest.raises(RestInvokerException) as e:
            rest_adapter.get("fake url")
        assert e is not None

    def test_get_returns_type_response(self, rest_adapter):
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
