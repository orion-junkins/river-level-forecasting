from abc import ABC
import os

from rlf.forecasting.catchment_data import CatchmentData


DEFAULT_WORK_DIR = os.path.join("trained_models")


class BaseForecaster(ABC):
    """Abstract base class for Forecaster objects. Contains Forecaster attributes and methods which are generic across Training and Inference."""

    def __init__(
        self,
        catchment_data: CatchmentData,
        root_dir: str = DEFAULT_WORK_DIR,
        filename: str = "frcstr",
        scaler_filename: str = "scaler",
    ) -> None:
        """Creates a Forecaster instance. Inheriting classes are expected to call this init before performing their specialized init functionality.

        Args:
            catchment_data (CatchmentData): All needed data about the catchment. Defaults to None.
            root_dir (str, optional): The working directory for model saving. Defaults to DEFAULT_WORK_DIR.
            filename (str, optional): The specific filename under which this forecaster should save/load its core ensemblke model. Defaults to "frcstr".
            scaler_filename (str, optional): The specific filename under which the scalers will be saved to and loaded from. Defaults to "scaler".

        Raises:
            ValueError: Raises error if there is already a file with filename or scaler_filename within root_dir.
        """
        self.catchment_data = catchment_data
        self.filename = filename
        self.scaler_filename = scaler_filename
        self.root_dir = root_dir
        self.work_dir = os.path.join(root_dir, self.catchment_data.name)

    @property
    def name(self) -> str:
        """The name of the catchment.

        Returns:
            str: The name of the catchment as provided by the underlying CatchmentData instance.
        """
        return self.catchment_data.name

    @property
    def model_save_path(self) -> str:
        """The full path to which the ensemble will be saved/loaded.

        Returns:
            str: The path of the saved ensemble model.
        """
        return os.path.join(self.work_dir, self.filename)

    @property
    def scaler_save_path(self) -> str:
        """The full path to which the scalers will be saved/loaded.

        Returns:
            str: The path of the saved scalers.
        """
        return os.path.join(self.work_dir, self.scaler_filename)

    @property
    def num_tributary_models(self) -> int:
        """The number of tributary models (i.e. the number of data locations).

        Returns:
           int: The number of datasets.
        """
        return self.catchment_data.num_weather_datasets
