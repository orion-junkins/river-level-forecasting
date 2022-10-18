import os
from abc import ABC

DEFAULT_WORK_DIR = os.path.join("trained_models")


class Forecaster_ABC(ABC):
    def __init__(self, catchment_data=None, root_dir=DEFAULT_WORK_DIR, filename="frcstr") -> None:
        self.catchment_data = catchment_data
        self.work_dir = os.path.join(root_dir, self.name)
        self.ensemble_save_path = os.path.join(self.work_dir, filename + ".pkl")

        os.makedirs(self.work_dir, exist_ok=True)
        if (os.path.isfile(self.ensemble_save_path)):
            raise ValueError(self.ensemble_save_path + " already exists. Specify a unique save path.")

    @property
    def name(self):
        return self.catchment_data.name

    @property
    def ensemble_save_path(self):
        return os.path.join(self.work_dir, self.name)
    
    @property
    def num_tributary_models(self):
        return self.catchment_data.num_data_sets
