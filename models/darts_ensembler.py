import functools
from typing import Callable, List

from darts import TimeSeries
from darts.models.forecasting.regression_ensemble_model import RegressionEnsembleModel


def _columns_to_drop(all_columns: List[str], columns_to_keep: List[str]) -> List[str]:
    columns_to_keep = set(columns_to_keep)
    return [c for c in all_columns if c not in columns_to_keep]


def _modify_covariates(covariates, columns_to_keep):
    if isinstance(covariates, TimeSeries):
        covariates = covariates.drop_columns(_columns_to_drop(covariates.columns, columns_to_keep))
    elif covariates is not None:
        covariates = [covariate.drop_columns(_columns_to_drop(covariate.columns, columns_to_keep)) for covariate in covariates]
    return covariates


def _wrap_with_column_selection(original_method: Callable, columns_to_keep: List[str]) -> Callable:
    def wrapper(*args, **kwargs):
        if "past_covariates" in kwargs:
            kwargs["past_covariates"] = _modify_covariates(kwargs["past_covariates"], columns_to_keep)
        if "future_covariates" in kwargs:
            kwargs["future_covariates"] = _modify_covariates(kwargs["future_covariates"], columns_to_keep)
        
        return original_method(*args, **kwargs)
    wrapper = functools.update_wrapper(wrapper, original_method)
    return wrapper


class ColumnDroppingEnsemblingModel:
    def __init__(self, tributary_models, columns_for_models):
        for model, columns_to_keep in zip(tributary_models, columns_for_models):
            model.fit = _wrap_with_column_selection(model.fit, columns_to_keep)
            model.predict = _wrap_with_column_selection(model.predict, columns_to_keep)
        
        self.tributary_models = tributary_models
        self.ensembler = RegressionEnsembleModel(tributary_models, 10)
    
    def fit(self, training_series, training_covariates):
        return self.ensembler.fit(training_series, past_covariates=training_covariates)

