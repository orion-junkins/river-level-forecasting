import requests
from forecasting.data_fetching_utilities.api.exceptions import RestInvokerException
from forecasting.data_fetching_utilities.api.models import Response


class RestInvoker():

    def __init__(self, protocol: str = None, hostname: str = None, version: str = None, ssl_verify: bool = True):
        self._protocol = protocol
        self._hostname = hostname
        self._version = version
        self._ssl_verify = ssl_verify

    def _apiCall(self, method: str, path: str, parameters: dict = None, data: dict = None) -> Response:

        url: str = f"{self._protocol}://"
        if self._hostname is not None:
            url += f"{self._hostname}/"
        if self._version is not None:
            url += f"{self._version}/"
        if path is not None:
            url += f"{path}"

        try:
            response = requests.request(
                method=method, url=url, verify=self._ssl_verify, params=parameters, json=data)
        except requests.exceptions.RequestException as e:
            raise RestInvokerException("Request failed") from e
        return Response(status_code=response.status_code, url=response.url, message=response.reason, headers=response.headers, data=response.json())

    def get(self, path: str, parameters: dict = None) -> Response:
        return self._apiCall(method="GET", path=path, parameters=parameters)
