from dataclasses import dataclass
from typing import Optional


@dataclass
class Response():
    """Response object from a REST API

    Args:
        status_code (int): The status code of the response
        url (str): The URL of the response
        message (str): The message of the response
        headers (dict): The headers of the response
        data (dict, optional): The data of the response. Defaults to None.
    """
    status_code: int
    url: str
    message: str
    headers: dict
    data: Optional[dict] = None
