import logging
import pickle
from typing import Callable, List, Optional, Set, Tuple, Union

from darts import TimeSeries
from darts.models.forecasting.forecasting_model import GlobalForecastingModel
from darts.utils import _build_tqdm_iterator
from darts.utils.timeseries_generation import generate_index
from darts.utils.utils import (
    drop_after_index,
    drop_before_index,
    series2seq,
)
import numpy as np
import pandas as pd

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

    def historical_forecasts(
        self,
        series: CovariateType,
        past_covariates: Optional[CovariateType] = None,
        future_covariates: Optional[CovariateType] = None,
        num_samples: int = 1,
        train_length: Optional[int] = None,
        start: Optional[Union[pd.Timestamp, float, int]] = None,
        forecast_horizon: int = 1,
        stride: int = 1,
        retrain: Union[bool, int, Callable[..., bool]] = True,
        overlap_end: bool = False,
        last_points_only: bool = True,
        verbose: bool = False,
        show_warnings: bool = False
    ) -> Union[TimeSeries, List[TimeSeries], CovariateType]:
        past_covariates, future_covariates = self._preprocess_input_data(past_covariates, future_covariates)

        # we will never retrain the model and have removed the functionality to do so
        assert retrain is False

        model = self._base_model

        assert model._fit_called

        series = series2seq(series)
        past_covariates = series2seq(past_covariates)
        future_covariates = series2seq(future_covariates)

        if len(series) == 1:
            # Use tqdm on the outer loop only if there's more than one series to iterate over
            # (otherwise use tqdm on the inner loop).
            outer_iterator = series
        else:
            outer_iterator = _build_tqdm_iterator(series, verbose)

        forecasts_list = []
        for idx, series_ in enumerate(outer_iterator):
            past_covariates_ = past_covariates[idx] if past_covariates else None
            future_covariates_ = future_covariates[idx] if future_covariates else None

            # Prediction
            historical_forecasts_time_index_predict = (
                model._get_historical_forecastable_time_index(
                    series_,
                    past_covariates_,
                    future_covariates_,
                    is_training=False,
                )
            )

            if historical_forecasts_time_index_predict is None:
                raise ValueError(
                        "Cannot build a single input for prediction with the provided model, "
                        f"`series` and `*_covariates` at series index: {idx}. The minimum "
                        "prediction input time index requirements were not met. "
                        "Please check the time index of `series` and `*_covariates`."
                    )

            historical_forecasts_time_index = (
                    historical_forecasts_time_index_predict
                )

            # Take into account overlap_end, and forecast_horizon.
            last_valid_pred_time = model._get_last_prediction_time(
                series_,
                forecast_horizon,
                overlap_end,
            )

            # The historical_forecasts_time_index end (which was just model dependent so far) is readjusted
            # by function parameters overlap_end and forecast_horizon.
            historical_forecasts_time_index = drop_after_index(
                historical_forecasts_time_index, last_valid_pred_time
            )

            # adjust maximum index with optional `start` value
            if start is not None:
                start_time_ = series_.get_timestamp_at_point(start)
                if (
                    not historical_forecasts_time_index[0]
                    <= start_time_
                    <= historical_forecasts_time_index[-1]
                ):
                    if show_warnings:
                        if not isinstance(start, pd.Timestamp):
                            start_value_msg = f"value `{start}` corresponding to timestamp `{start_time_}`"
                        else:
                            start_value_msg = f"time `{start_time_}`"

                        if start_time_ < historical_forecasts_time_index[0]:
                            logging.warning(
                                f"`start` {start_value_msg} is before the first predictable/trainable historical "
                                f"forecasting point for series at index: {idx}. Ignoring `start` for this series and "
                                f"beginning at first trainable/predictable time: {historical_forecasts_time_index[0]}. "
                                f"To hide these warnings, set `show_warnings=False`."
                            )
                        else:
                            logging.warning(
                                f"`start` {start_value_msg} is after the last trainable/predictable historical "
                                f"forecasting point for series at index: {idx}. This would results in empty historical "
                                f"forecasts. Ignoring `start` for this series and beginning at first trainable/"
                                f"predictable time: {historical_forecasts_time_index[0]}. Non-empty forecasts can be "
                                f"generated by setting `start` value to times between (including): "
                                f"{historical_forecasts_time_index[0], historical_forecasts_time_index[-1]}. "
                                f"To hide these warnings, set `show_warnings=False`."
                            )
                    # ignore user-defined `start`
                    start_time_ = None

                if start_time_ is not None:
                    historical_forecasts_time_index = drop_before_index(
                        historical_forecasts_time_index,
                        start_time_,
                    )

            if len(series) == 1:
                # Only use tqdm if there's no outer loop
                iterator = _build_tqdm_iterator(
                    historical_forecasts_time_index[::stride], verbose
                )
            else:
                iterator = historical_forecasts_time_index[::stride]

            # Either store the whole forecasts or only the last points of each forecast, depending on last_points_only
            forecasts = []

            last_points_times = []
            last_points_values = []

            # iterate and forecast
            for pred_time in iterator:
                train_series = series_.drop_after(pred_time)

                # for regression models with lags=None, lags_past_covariates=None and min(lags_future_covariates)>=0,
                # the first predictable timestamp is the first timestamp of the series, a dummy ts must be created
                # to support `predict()`
                # >>>> I don't think this is needed but I haven't grokked what is happening yet
                # if len(train_series) == 0:
                #     train_series = TimeSeries.from_times_and_values(
                #         times=generate_index(
                #             start=pred_time - 1 * series_.freq,
                #             length=1,
                #             freq=series_.freq,
                #         ),
                #         values=np.array([np.NaN]),
                #     )

                forecast = model._predict_wrapper(
                    n=forecast_horizon,
                    series=train_series,
                    past_covariates=past_covariates_,
                    future_covariates=future_covariates_,
                    num_samples=num_samples,
                    verbose=verbose,
                )

                if last_points_only:
                    last_points_values.append(forecast.all_values(copy=False)[-1])
                    last_points_times.append(forecast.end_time())
                else:
                    forecasts.append(forecast)

            if last_points_only:
                forecasts_list.append(
                    TimeSeries.from_times_and_values(
                        generate_index(
                            start=last_points_times[0],
                            end=last_points_times[-1],
                            freq=series_.freq * stride,
                        ),
                        np.array(last_points_values),
                        columns=series_.columns,
                        static_covariates=series_.static_covariates,
                        hierarchy=series_.hierarchy,
                    )
                )
            else:
                forecasts_list.append(forecasts)

        return forecasts_list if len(series) > 1 else forecasts_list[0]

    @property
    def input_chunk_length(self) -> int:
        return self._base_model.input_chunk_length

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

    def _modify_covariates(self, covariates: Optional[CovariateType]) -> Optional[CovariateType]:
        if isinstance(covariates, TimeSeries):
            covariates = covariates.drop_columns(self._columns_to_drop(covariates.columns))
        elif covariates is not None:
            covariates = [covariate.drop_columns(self._columns_to_drop(covariate.columns)) for covariate in covariates]
        return covariates

    def _columns_to_drop(self, all_columns: List[str]) -> List[str]:
        columns_to_keep = self._find_columns_to_keep(all_columns)
        return [c for c in all_columns if c not in columns_to_keep]

    def _find_columns_to_keep(self, all_columns: List[str]) -> Set[str]:
        # mypy is complaining that _column_prefix could be None but there is a guard to prevent that
        return {c for c in all_columns if c.startswith(self._column_prefix)}  # type: ignore[arg-type]

    def _model_encoder_settings(self) -> Tuple[int, int, bool, bool]:
        return self._base_model._model_encoder_settings()

    def save(self, path: str) -> None:
        """Save this contributing model to disk.

        This method will save the contributing model in two pieces.
        The first one, which will be named path, will be this contributing model object except for its base model.
        The base model's class will be saved with the contributing model instead.
        The second piece will be the actual base model.
        The base model's save method will be called with the suffix "_base_model" appended to path.
        Note, the base model will likely save itself in two seperate pieces itself.

        Args:
            path (str): Path to save this model to. Must include the intended filename.
        """
        with open(path, "wb") as f:
            base_model = self._base_model
            self._base_model = self._base_model.__class__
            pickle.dump(self, f)
            self._base_model = base_model

        self._base_model.save(path + "_base_model")

    @classmethod
    def load(cls, path: str, load_cpu: bool = False) -> "ContributingModel":
        """Load a contributing model object from disk.

        Loading a contributing model involves first unpickling the ContributingModel object located at path.
        Next, the base model will be loaded by invoking the load class method of the base model.
        This method assumes that the naming conventions of ContributingModel.save is being used.

        Args:
            path (str): Path to a file containing a pickled ContributingModel.
            load_cpu (bool): If True then when loading the base model set it to run inference on CPU. Defaults to False.

        Returns:
            ContributingModel: Fully loaded ContributingModel
        """
        with open(path, "rb") as f:
            contributing_model = pickle.load(f)

        if load_cpu:
            contributing_model._base_model = contributing_model._base_model.load(path + "_base_model", map_location="cpu")
            contributing_model._base_model.to_cpu()
        else:
            contributing_model._base_model = contributing_model._base_model.load(path + "_base_model")

        return contributing_model

    def untrained_model(self) -> "ContributingModel":
        """Return an untrained model of the same type as this model.

        Returns:
            ContributingModel: Untrained model of the same type as this model.
        """
        return ContributingModel(self._base_model.untrained_model(), self._column_prefix)
