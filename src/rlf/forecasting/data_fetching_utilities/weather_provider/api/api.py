from rlf.forecasting.data_fetching_utilities.weather_provider.api.api_adapter_abc import BaseAPIAdapter
from rlf.forecasting.data_fetching_utilities.weather_provider.api.models import Response
from rlf.forecasting.data_fetching_utilities.weather_provider.api.rest_invoker import RestInvoker


class RequestBuilder():
    """Build the request payload
    """

    def __init__(self, api_adapter: BaseAPIAdapter,
                 ssl_verify: bool = True,
                 protocol: str = "https",
                 hostname: str = "archive-api.open-meteo.com",
                 version: str = "v1",
                 path: str = "era5",
                 parameters: dict = {}) -> None:
        """Builds the request payload for any APIAdapter object that contains the get_payload() method

        Args:
            api_adapter (APIAdapter): An APIAdapter object that contains the get_payload() method
            ssl_verify (bool, optional): Option to verify the SSL certificate. Defaults to True.
            protocol (str, optional): The protocol to use. Defaults to "https".
            hostname (str, optional): The hostname to use. Defaults to "archive-api.open-meteo.com".
            version (str, optional): The version of the API to use. Defaults to "v1".
            path (str, optional): The path to use. Defaults to "era5".
            parameters (dict, optional): The parameters to use. Defaults to None.
        """

        self.api_adapter = api_adapter
        self.ssl_verify = ssl_verify
        self.protocol = protocol
        self.hostname = hostname
        self.version = version
        self.path = path
        self.parameters = parameters

    def get(self) -> Response:
        """Get the response from a REST API

        Returns:
            Response: The response object from the REST API containing response body, headers, status code
        """
        self.parse_payload()

        rest_invoker = RestInvoker(
            protocol=self.protocol,
            hostname=self.hostname,
            version=self.version,
            ssl_verify=self.ssl_verify)

        return rest_invoker.get(path=self.path, parameters=self.parameters)

    def parse_payload(self) -> None:
        """Parse the payload from the APIAdapter object and set the parameters attribute"""
        payload = self.get_payload()
        self.protocol = payload["protocol"]
        self.hostname = payload["hostname"]
        self.version = payload["version"]
        self.parameters = payload["parameters"]

    def get_payload(self) -> dict:
        return self.api_adapter.get_payload()
