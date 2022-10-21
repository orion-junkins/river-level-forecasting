import requests
from rlf.forecasting.data_fetching_utilities.weather_provider.api.exceptions import RestInvokerException
from rlf.forecasting.data_fetching_utilities.weather_provider.api.models import Response


class RestInvoker():
    """Invoke a REST API
    """

    def __init__(self, protocol: str = None, hostname: str = None, version: str = None, ssl_verify: bool = True):
        """Invoke a REST API using the requests library

        Args:
            protocol (str, optional): The protocol to use. Defaults to None.
            hostname (str, optional): The hostname to use. Defaults to None.
            version (str, optional): The version to use. Defaults to None.
            ssl_verify (bool, optional):  Option to verify the SSL certificate. Defaults to True.
        """
        self._protocol = protocol
        self._hostname = hostname
        self._version = version
        self._ssl_verify = ssl_verify

    def _apiCall(self, method: str, path: str, parameters: dict = None, data: dict = None) -> Response:
        """Perform an HTTP request

        Args:
            method (str): The HTTP method to use
            path (str): The path to use
            parameters (dict, optional): The parameters to use. Defaults to None.
            data (dict, optional): The data to use. Defaults to None.

        Raises:
            RestInvokerException: If the request fails

        Returns:
            Response: The response object from the REST API containing response body, headers, status code
        """
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
        """Perform a GET request

        Args:
            path (str): The path to use
            parameters (dict, optional): The parameters to use. Defaults to None.

        Returns:
            Response: The response object from the REST API containing response body, headers, status code
        """
        return self._apiCall(method="GET", path=path, parameters=parameters)
