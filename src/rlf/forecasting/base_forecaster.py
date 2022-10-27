import os
from abc import ABC

from rlf.forecasting.catchment_data import CatchmentData
from rlf.forecasting.base_dataset import BaseDataset


DEFAULT_WORK_DIR = os.path.join("trained_models")


class BaseForecaster(ABC):
    """Abstract base class for Forecaster objects. Contains Forecaster attributes and methods which are generic across Training and Inference.
    """
    def __init__(self, catchment_data: CatchmentData = None, dataset: BaseDataset = None, root_dir: str = DEFAULT_WORK_DIR, filename: str = "frcstr") -> None:
        """Creates a Forecaster instance. Inheriting classes are expected to call this init before performing their specialized init functionality.

        Args:
            catchment_data (CatchmentData, optional): All needed data about the catchment. Defaults to None.
            dataset (BaseDataset, optional): Dataset object for training OR inference. Must align with subtype (ie TrainingDataset for TrainingForecaster).
            root_dir (str, optional): The working directory for model saving. Defaults to DEFAULT_WORK_DIR.
            filename (str, optional): The specific filename under which this forecaster should save/load its core ensemblke model. Defaults to "frcstr".

        Raises:
            ValueError: Raises error if there is already a file with filename within root_dir.
        """
        self.catchment_data = catchment_data
        self.dataset = dataset
        self.filename = filename
        self.work_dir = os.path.join(root_dir, self.catchment_data.name)

        os.makedirs(self.work_dir, exist_ok=True)
        if (os.path.isfile(self.ensemble_save_path)):
            raise ValueError(self.ensemble_save_path + " already exists. Specify a unique save path.")

    @property
    def name(self) -> str:
        """The name of the catchment.

        Returns:
            str: The name of the catchment as provided by the underlying CatchmentData isntance.
        """
        return self.catchment_data.name

    @property
    def ensemble_save_path(self) -> str:
        """The full path to which the ensemble will be saved/loaded.

        Returns:
            str: The path of the saved ensemble model.
        """
        return os.path.join(self.work_dir, self.name, self.filename)

    @property
    def num_tributary_models(self) -> int:
        """The number of tributrary models (ie the number of data locations).

        Returns:
           int: The number of datasets.
        """
        return self.catchment_data.num_weather_datasets
