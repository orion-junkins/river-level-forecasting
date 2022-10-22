class Response():
    """Response object from a REST API
    """

    def __init__(self, status_code: int, url: str, message: str, headers: dict, data: list[dict] = None):
        """Response object from a REST API

        Args:
            status_code (int): The status code of the response
            url (str): The URL of the response
            message (str): The message of the response
            headers (dict): The headers of the response
            data (list[dict], optional): The data of the response. Defaults to None.
        """
        self.status_code = status_code
        self.url = url
        self.message = message
        self.headers = headers
        self.data = data if data is not None else []
