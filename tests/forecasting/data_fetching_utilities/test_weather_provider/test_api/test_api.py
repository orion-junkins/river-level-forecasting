import pytest

from rlf.forecasting.data_fetching_utilities.open_meteo.open_meteo_adapter import OpenMeteoAdapter
from rlf.forecasting.data_fetching_utilities.api.api import RequestBuilder
from rlf.forecasting.data_fetching_utilities.api.models import Response


class TestRequestBuilder:

    @pytest.fixture
    def open_meteo_adapter(self):
        return OpenMeteoAdapter()

    @pytest.fixture
    def request_builder(self, open_meteo_adapter):
        return RequestBuilder(open_meteo_adapter)

    def test_init(self, request_builder):
        try:
            request_builder
        except Exception:
            pytest.fail("Failed to initialize RequestBuilder")

    def test_has_attributes(self, request_builder):
        assert hasattr(request_builder, "ssl_verify")
        assert hasattr(request_builder, "protocol")
        assert hasattr(request_builder, "hostname")
        assert hasattr(request_builder, "version")
        assert hasattr(request_builder, "path")
        assert hasattr(request_builder, "parameters")

    def test_has_api_adapter(self, request_builder):
        assert request_builder.api_adapter is not None

    def test_has_get_method(self, request_builder):
        assert hasattr(request_builder, "get")

    def test_get_returns_response(self, request_builder):
        response = request_builder.get()
        assert isinstance(response, Response)

    def test_get_returns_response_with_status_code(self, request_builder):
        response = request_builder.get()
        assert response.status_code is not None

    def test_get_returns_response_with_content(self, request_builder):
        response = request_builder.get()
        assert response.data is not None

    def test_has_parse_payload_method(self, request_builder):
        assert hasattr(request_builder, "parse_payload")

    def test_has_get_payload_method(self, request_builder):
        assert hasattr(request_builder, "get_payload")

    def test_api_adapter_can_get_payload(self, open_meteo_adapter):
        payload = open_meteo_adapter.get_payload()
        assert payload is not None
        assert isinstance(payload, dict)

    def test_parse_payload_sets_attributes_same_as_incoming_adapter(self, request_builder, open_meteo_adapter):
        request_builder.parse_payload()
        assert request_builder.protocol == open_meteo_adapter.protocol
        assert request_builder.hostname == open_meteo_adapter.hostname
        assert request_builder.version == open_meteo_adapter.version
        assert request_builder.parameters == open_meteo_adapter.get_parameters()
