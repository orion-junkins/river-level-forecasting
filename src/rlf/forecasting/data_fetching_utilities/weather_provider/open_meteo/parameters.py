
HOURLY_PARAMETERS = ["temperature_2m"]


def get_hourly_parameters() -> list:
    """The parameters to use in a request used by the Open Meteo API

    Returns:
        list: The hourly parameters to use in a request which are weather measurements
    """
    return HOURLY_PARAMETERS
