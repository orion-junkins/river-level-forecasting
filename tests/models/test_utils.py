from darts.models.forecasting.regression_ensemble_model import RegressionEnsembleModel
from darts.models.forecasting.rnn_model import RNNModel
from pytorch_lightning import Trainer

from rlf.models.contributing_model import ContributingModel
from rlf.models.utils import repair_regression_ensemble_model


def test_repair_contributing_model():
    ensemble_model = RegressionEnsembleModel([ContributingModel(RNNModel(1))], 1)
    ensemble_model.models[0]._base_model.trainer = None

    repair_regression_ensemble_model(ensemble_model)

    assert isinstance(ensemble_model.models[0]._base_model.trainer, Trainer)


def test_repair_regular_model():
    ensemble_model = RegressionEnsembleModel([RNNModel(1)], 1)
    ensemble_model.models[0].trainer = None

    repair_regression_ensemble_model(ensemble_model)

    assert isinstance(ensemble_model.models[0].trainer, Trainer)
