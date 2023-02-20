import os

from darts.models.forecasting.regression_ensemble_model import RegressionEnsembleModel
from darts.models.forecasting.rnn_model import RNNModel
from pytorch_lightning import Trainer

from rlf.models.contributing_model import ContributingModel
from rlf.models.utils import repair_regression_ensemble_model, save_ensemble_model, load_ensemble_model


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


def test_save_ensemble_model(tmp_path):
    ensemble_model = RegressionEnsembleModel([ContributingModel(RNNModel(1))], 1)

    save_ensemble_model(tmp_path, ensemble_model)

    assert os.path.exists(f"{tmp_path}/ensemble")
    assert os.path.exists(f"{tmp_path}/contributing_model_0")
    assert os.path.exists(f"{tmp_path}/contributing_model_0_base_model")


def test_load_ensemble_model(tmp_path):
    ensemble_model = RegressionEnsembleModel([ContributingModel(RNNModel(1))], 1)
    ensemble_model.__test_tag = "ensemble_model"
    ensemble_model.models[0].__test_tag = "contributing_model"
    ensemble_model.models[0]._base_model.__test_tag = "base_model"

    save_ensemble_model(tmp_path, ensemble_model)
    loaded_model = load_ensemble_model(tmp_path)

    assert loaded_model.__test_tag == "ensemble_model"
    assert loaded_model.models[0].__test_tag == "contributing_model"
    assert loaded_model.models[0]._base_model.__test_tag == "base_model"
