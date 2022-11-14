from darts.models.forecasting.regression_ensemble_model import RegressionEnsembleModel
from pytorch_lightning import Trainer

from rlf.models.contributing_model import ContributingModel


def repair_regression_ensemble_model(model: RegressionEnsembleModel) -> None:
    """Repair RegressionEnsembleModel that has been unpickled.

    For some reason, certain models do not pickle correctly.
    This function attempts to repair RegressionEnsembleModels that are pickled incorrectly.
    Currently, the only known defect is that a proper Trainer object is not pickled/unpickled which is repaired by instantiating a new default one.

    Args:
        model (RegressionEnsembleModel): Model that needs to be repaired after being unpickled.
    """
    for contributing_model in model.models:
        if isinstance(contributing_model, ContributingModel):
            contributing_model._base_model.trainer = Trainer()
        else:
            contributing_model.trainer = Trainer()
