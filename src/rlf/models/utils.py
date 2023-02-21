import os

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


def save_ensemble_model(base_dir: str, model: RegressionEnsembleModel) -> None:
    """Save an ensemble model to disk.

    Args:
        base_dir (str): Base directory to save the ensemble model and its contributing models to.
        model (RegressionEnsembleModel): Ensemble model to save.
    """
    for i, contributing_model in enumerate(model.models):
        contributing_model.save(os.path.join(base_dir, f"contributing_model_{i}"))

    old_models = model.models
    model.models = [None] * len(model.models)

    model.save(os.path.join(base_dir, "frcstr"))

    model.models = old_models


def load_ensemble_model(base_dir: str) -> RegressionEnsembleModel:
    """Load an ensemble model from disk.

    Args:
        path (str): Base directory where the ensemble model and its contributing models are located.

    Returns:
        RegressionEnsembleModel: Loaded ensemble model.
    """
    model = RegressionEnsembleModel.load(os.path.join(base_dir, "frcstr"))
    contributing_models = [
        ContributingModel.load(os.path.join(base_dir, f"contributing_model_{i}"))
        for i in range(len(model.models))
    ]

    model.models = contributing_models

    return model
