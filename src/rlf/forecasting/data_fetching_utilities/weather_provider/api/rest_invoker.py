import requests
from typing import Optional

from rlf.forecasting.data_fetching_utilities.weather_provider.api.models import Response
from rlf.forecasting.data_fetching_utilities.weather_provider.api.exceptions import RestInvokerException


class RestInvoker():
    """Invoke a REST API
    """

    def __init__(self, protocol: Optional[str] = None, hostname: Optional[str] = None, version: Optional[str] = None, ssl_verify: bool = True) -> None:
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

    def _apiCall(self, method: str, path: str, parameters: Optional[dict] = None, data: Optional[dict] = None) -> Response:
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
                method=method,
                url=url,
                verify=self._ssl_verify,
                params=parameters,
                json=data,
                timeout=5,
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.0.0 Safari/537.36 Edg/107.0.1418.52"})
            if response.status_code == 200:
                return Response(status_code=response.status_code, url=response.url, message=response.reason, headers=response.headers, data=response.json())
            else:
                raise RestInvokerException(
                    f"Error calling the API: {response.reason} ({response.status_code}) \n {response.json()}")

        except requests.exceptions.RequestException as e:
            raise RestInvokerException("Error: {}".format(e)) from e

    def get(self, path: str, parameters: Optional[dict] = None) -> Response:
        """Perform a GET request

        Args:
            path (str): The path to use
            parameters (dict, optional): The parameters to use. Defaults to None.

        Returns:
            Response: The response object from the REST API containing response body, headers, status code
        """
        return self._apiCall(method="GET", path=path, parameters=parameters)
