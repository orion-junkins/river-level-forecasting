from abc import ABC, abstractmethod


class APIAdapterABC(ABC):
    """Abstract base class for APIAdapter objects"""

    def __init__(self, protocol: str, hostname: str, version: str, path: str, parameters: list[str]) -> None:
        """Initialize the APIAdapter object

        Args:
            protocol (str): The protocol to use
            hostname (str): The hostname to use
            version (str): The version to use
            path (str): The path to use
            parameters (list[str]): The parameters to use
        """
        self.protocol = protocol
        self.hostname = hostname
        self.version = version
        self.path = path
        self.parameters = parameters

    @abstractmethod
    def get_payload(self) -> dict:
        """Get the payload for the request in the form of a hash map

        Returns:
            dict: The payload for the request
        """
        return NotImplementedError
