import os
from abc import ABC

DEFAULT_WORK_DIR = os.path.join("trained_models")


class Forecaster_ABC(ABC):
    """Abstract base class for Forecaster objects. Contains Forecaster attributes and methods which are generic across Training and Inference.
    """
    def __init__(self, catchment_data=None, root_dir=DEFAULT_WORK_DIR, filename="frcstr") -> None:
        """Creates a Forecaster instance. Inheriting classes are expected to call this init before performing their specialized init functionality.

        Args:
            catchment_data (CatchmentData, optional): All needed data about the catchment. Defaults to None.
            root_dir (_type_, optional): The working directory for model saving. Defaults to DEFAULT_WORK_DIR.
            filename (str, optional): The specific filename under which this forecaster should save/load its core ensemblke model. Defaults to "frcstr".

        Raises:
            ValueError: Raises error if there is already a file with filename within root_dir.
        """
        self.catchment_data = catchment_data
        self.work_dir = os.path.join(root_dir, self.name)
        self.ensemble_save_path = os.path.join(self.work_dir, filename + ".pkl")

        os.makedirs(self.work_dir, exist_ok=True)
        if (os.path.isfile(self.ensemble_save_path)):
            raise ValueError(self.ensemble_save_path + " already exists. Specify a unique save path.")

    @property
    def name(self):
        """The name of the catchment.

        Returns:
            str: The name of the catchment as provided by the underlying CatchmentData isntance.
        """
        return self.catchment_data.name

    @property
    def ensemble_save_path(self):
        """The full path to which the ensemble will be saved/loaded.

        Returns:
            os.path: The path of the saved ensemble model.
        """
        return os.path.join(self.work_dir, self.name)

    @property
    def num_tributary_models(self):
        """The number of tributrary models (ie the number of data locations).

        Returns:
           int: The number of datasets.
        """
        return self.catchment_data.num_weather_datasets
