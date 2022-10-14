from forecasting.data_fetching_utilities.api.rest_invoker import RestInvoker
from forecasting.data_fetching_utilities.api.models import APIAdapter


class RequestBuilder():

    def __init__(self, api_adapter: APIAdapter,
                 ssl_verify: bool = True,
                 protocol: str = "https",
                 hostname: str = "archive-api.open-meteo.com",
                 version: str = "v1",
                 path: str = "era5",
                 parameters: dict = None):

        self.api_adapter = api_adapter
        self.ssl_verify = ssl_verify
        self.protocol = protocol
        self.hostname = hostname
        self.version = version
        self.path = path
        self.parameters = parameters

    def get(self):
        self.parse_payload()

        rest_invoker = RestInvoker(
            protocol=self.protocol,
            hostname=self.hostname,
            version=self.version,
            ssl_verify=self.ssl_verify)

        return rest_invoker.get(path=self.path, parameters=self.parameters)

    def parse_payload(self) -> None:
        payload = self.get_payload()
        self.protocol = payload["protocol"]
        self.hostname = payload["hostname"]
        self.version = payload["version"]
        self.parameters = payload["parameters"]

    def get_payload(self) -> dict:
        return self.api_adapter.get_payload()
