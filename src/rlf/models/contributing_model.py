from typing import List, Optional, Set, Sequence, Tuple

from darts import TimeSeries
from darts.models.forecasting.forecasting_model import GlobalForecastingModel

from rlf.types import CovariateType


class ContributingModel(GlobalForecastingModel):
    """ContributingModel wraps another forecasting model that is used to contribute to the ensembled prediction."""

    def __init__(
        self,
        base_model: GlobalForecastingModel,
        column_prefix: Optional[str] = None,
    ) -> None:
        """Initialize ContributingModel class.

        Args:
            base_model (GlobalForecastingModel): Base Darts model that this tributary model wraps around.
            column_prefix (str, optional): If given then only columns in past_covariates or future_covariates that start with column_prefix will be passed to the base model. If None then all columns will be passed to the base model. Defaults to None.
        """
        self._base_model = base_model
        self._column_prefix = column_prefix

    def fit(
        self,
        *,
        series: TimeSeries,
        past_covariates: Optional[CovariateType] = None,
        future_covariates: Optional[CovariateType] = None,
        **kwargs,
    ) -> GlobalForecastingModel:
        """Fit the base model to given series and covariates.

        Args:
            series (TimeSeries): Target data to use for fitting.
            past_covariates (CovariateType): Past covariates to use for training. See Darts documentation for information regarding this parameter.
            future_covariates (CovariateType): Future covariates to use for training. See Darts documentation for information regarding this parameter.

        Returns:
            GlobalForecastingModel: self
        """
        past_covariates, future_covariates = self._preprocess_input_data(past_covariates, future_covariates)
        self._base_model.fit(series=series, past_covariates=past_covariates, future_covariates=future_covariates, **kwargs)
        return self

    def predict(
        self,
        n: int,
        series: Optional[CovariateType] = None,
        past_covariates: Optional[CovariateType] = None,
        future_covariates: Optional[CovariateType] = None,
        **kwargs,
    ) -> TimeSeries:
        """Predict n points using the supplied data.

        Args:
            n (int): Number of points to predict.
            series (CovariateType): Series data to pass to base model. See Darts documentation for information regarding this parameter.
            past_covariates (CovariateType): Past covariates to use for prediction. See Darts documentation for information regarding this parameter.
            future_covariates (CovariateType): Future covariates to use for prediction. See Darts documentation for information regarding this parameter.

        Returns:
            TimeSeries: n predictions of river levels.
        """
        past_covariates, future_covariates = self._preprocess_input_data(past_covariates, future_covariates)
        return self._base_model.predict(n=n, series=series, past_covariates=past_covariates, future_covariates=future_covariates, **kwargs)

    @property
    def _fit_called(self) -> bool:
        return self._base_model._fit_called

    def _preprocess_input_data(
        self,
        past_covariates: Optional[CovariateType] = None,
        future_covariates: Optional[CovariateType] = None
    ) -> Tuple[CovariateType, CovariateType]:
        """If a column prefix is being used then return a filtered set of columns otherwise return the covariates.

        Args:
            past_covariates (CovariateType): Covariates to filter.
            future_covariates (CovariateType): Covariates to filter.

        Returns:
            CovariateType, CovariateType: Filtered past_covariates and future_covariates in that order.
        """
        if self._column_prefix is not None:
            past_covariates = self._modify_covariates(past_covariates)
            future_covariates = self._modify_covariates(future_covariates)
        return past_covariates, future_covariates

    def _modify_covariates(self, covariates: TimeSeries | Sequence[TimeSeries] | None) -> TimeSeries | Sequence[TimeSeries] | None:
        if isinstance(covariates, TimeSeries):
            covariates = covariates.drop_columns(self._columns_to_drop(covariates.columns))
        elif covariates is not None:
            covariates = [covariate.drop_columns(self._columns_to_drop(covariate.columns)) for covariate in covariates]
        return covariates

    def _columns_to_drop(self, all_columns: List[str]) -> List[str]:
        columns_to_keep = self._find_columns_to_keep(all_columns)
        return [c for c in all_columns if c not in columns_to_keep]

    def _find_columns_to_keep(self, all_columns: List[str]) -> Set[str]:
        return {c for c in all_columns if c.startswith(self._column_prefix)}

    def _model_encoder_settings(self) -> Tuple[int, int, bool, bool]:
        return self._base_model._model_encoder_settings()
