import pytest
from forecasting.data_fetching_utilities.historical_weather.url_builder import UrlBuilder
from forecasting.data_fetching_utilities.historical_weather.weather_api_parameters import WeatherAPIParameters


class TestQueryBuilder:

    @pytest.fixture
    def url_builder(self):
        return UrlBuilder()

    def test_has_protocol(self, url_builder):
        assert hasattr(url_builder, "protocol")

    def test_has_host(self, url_builder):
        assert hasattr(url_builder, "host")

    def test_has_path(self, url_builder):
        assert hasattr(url_builder, "path")

    def test_has_parameters(self, url_builder):
        assert hasattr(url_builder, "parameters")

    def test_set_protocol(self, url_builder):
        url_builder.set_protocol("fake https")
        assert url_builder.protocol == "fake https"

    def test_get_protocol(self, url_builder):
        url_builder.set_protocol("fake https")
        assert url_builder.get_protocol() == "fake https"

    def test_set_host(self, url_builder):
        url_builder.set_host("fake.host")
        assert url_builder.host == "fake.host"

    def test_get_host(self, url_builder):
        url_builder.set_host("fake.host")
        assert url_builder.get_host() == "fake.host"

    def test_set_path(self, url_builder):
        url_builder.set_path("fake/path")
        assert url_builder.path == "fake/path"

    def test_get_path(self, url_builder):
        url_builder.set_path("fake/path")
        assert url_builder.get_path() == "fake/path"

    def test_set_parameters(self, url_builder):
        weather_api_parameters = WeatherAPIParameters()
        url_builder.set_parameters(weather_api_parameters)
        assert url_builder.parameters == weather_api_parameters

    def test_build(self, url_builder):
        url_builder.set_protocol("fake_https")
        url_builder.set_host("fakehost.com")
        url_builder.set_path("/fake/path")

        parameters = url_builder.get_parameters()
        query = parameters.build_query_string()
        assert url_builder.build() == "fake_https://fakehost.com/fake/path?" + query
        # fake_https://fakehost.com/fake/path?latitude=44.06&longitude=-121.31&start_date=2022-10-11&end_date=2022-10-11
