from typing import TypeVar

APIAdapter = TypeVar('APIAdapter', covariant=True)


class Response():

    def __init__(self, status_code: int, url: str, message: str, headers: dict, data: list[dict] = None):
        self.status_code = status_code
        self.url = url
        self.message = message
        self.headers = headers
        self.data = data if data is not None else []
