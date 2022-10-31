import pytest

from rlf.forecasting.data_fetching_utilities.weather_provider.api.exceptions import RestInvokerException
from rlf.forecasting.data_fetching_utilities.weather_provider.api.models import Response
from rlf.forecasting.data_fetching_utilities.weather_provider.api.rest_invoker import RestInvoker


@pytest.fixture
def rest_adapter():
    return RestInvoker(protocol="fake protocol", hostname="fake hostname", version="fake version", ssl_verify=True)


def test_get_fails(rest_adapter):
    with pytest.raises(RestInvokerException):
        rest_adapter.get("fake url")


def test_get_fails_with_message(rest_adapter):
    with pytest.raises(RestInvokerException):
        rest_adapter.get("fake url")


def test_get():
    invoker = RestInvoker(
        protocol="https", hostname="jsonplaceholder.typicode.com", version=None, ssl_verify=True)
    for i in range(4):
        res = invoker.get(path="posts")

    assert isinstance(res, Response)
    assert res.status_code == 200
    assert res.url == "https://jsonplaceholder.typicode.com/posts"
