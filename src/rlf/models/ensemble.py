from functools import reduce
import logging
import os
import pickle
from typing import Callable, List, Optional, Sequence, Union

from darts.timeseries import TimeSeries
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

from rlf.models.contributing_model import ContributingModel


class Ensemble(GlobalForecastingModel):

    def __init__(
            self,
            combiner: GlobalForecastingModel,
            contributing_models: List[ContributingModel],
            combiner_holdout_size: int,
            target_horizon: int = 12,
            combiner_train_stride: int = 25) -> None:
        """Initialize an ensemble.

        Args:
            combiner (GlobalForecastingModel): Model that is used to combine or aggregate the predictions of the contributing models to produce a single string of predictions.
            contributing_models (List[ContributingModel]): List of contributing models that will produce the intermediary predictions.
            combiner_holdout_size (int): The number of examples to hold out for training the combiner.
            target_horizon (int, optional): The target horizon to tune for. This represents the number of hours ahead from the current time that the model is specifically tuned for. Defaults to 12.
            combiner_train_stride (int, optional): Stride through the combiner hold out to use. This value is passed to historical_forecasts. Defaults to 25.
        """
        super().__init__()
        self.combiner = combiner
        self.contributing_models = contributing_models
        self._combiner_holdout_size = combiner_holdout_size
        self._target_horizon = target_horizon
        self._combiner_train_stride = combiner_train_stride

    def fit(self,
            series: TimeSeries,
            *,
            past_covariates: TimeSeries = None,
            future_covariates: TimeSeries = None,
            retrain_contributing_models: bool = False) -> "Ensemble":
        super().fit(series, past_covariates, future_covariates)
        contributing_model_y = series[:-self._combiner_holdout_size]

        combiner_start = len(series) - self._combiner_holdout_size + self.contributing_models[0].input_chunk_length

        for contributing_model in self.contributing_models:
            contributing_model.fit(series=contributing_model_y, past_covariates=past_covariates, future_covariates=future_covariates)

        del contributing_model_y

        predictions: List[TimeSeries] = [
            contributing_model.historical_forecasts(
                series=series,
                past_covariates=past_covariates,
                future_covariates=future_covariates,
                start=combiner_start,
                last_points_only=True,
                retrain=False,
                forecast_horizon=self._target_horizon,
                stride=self._combiner_train_stride,
                verbose=False,
                show_warnings=False
            )
            for contributing_model in self.contributing_models
        ]
        predictions = reduce(self._stack_op, predictions)

        self.combiner.fit(series=series.slice_intersect(predictions), future_covariates=predictions)

        del predictions

        if retrain_contributing_models:
            for contributing_model in self.contributing_models:
                contributing_model.fit(series=series, past_covariates=past_covariates, future_covariates=future_covariates)

        return self

    def fit_dataset(self, dataset, as_future: bool = True, retrain_contributing_models: bool = False):
        if as_future:
            return self.fit(
                dataset.y_train,
                future_covariates=dataset.X_train,
                retrain_contributing_models=retrain_contributing_models)
        else:
            return self.fit(
                dataset.y_train,
                past_covariates=dataset.X_train,
                retrain_contributing_models=retrain_contributing_models)

    def predict(self,
                n: int,
                series: TimeSeries,
                past_covariates: TimeSeries = None,
                future_covariates: TimeSeries = None,
                num_samples: int = 1,
                verbose: bool = False) -> TimeSeries:
        predictions = self._compute_and_stack_contributing_predictions(n,
                                                                       series,
                                                                       past_covariates=past_covariates,
                                                                       future_covariates=future_covariates)

        return self.combiner.predict(n, series=series, future_covariates=predictions, verbose=verbose)

    def historical_forecasts(
        self,
        series: Union[TimeSeries, Sequence[TimeSeries]],
        past_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        future_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        num_samples: int = 1,
        train_length: Optional[int] = None,
        start: Optional[Union[pd.Timestamp, float, int]] = None,
        forecast_horizon: int = 1,
        stride: int = 1,
        retrain: Union[bool, int, Callable[..., bool]] = True,
        overlap_end: bool = False,
        last_points_only: bool = True,
        verbose: bool = False,
        show_warnings: bool = True,
    ) -> Union[
        TimeSeries, List[TimeSeries], Sequence[TimeSeries], Sequence[List[TimeSeries]]
    ]:
        """Compute the historical forecasts that would have been obtained by this model on
        (potentially multiple) `series`.
        This method repeatedly builds a training set: either expanding from the beginning of `series` or moving with
        a fixed length `train_length`. It trains the model on the training set, emits a forecast of length equal to
        forecast_horizon, and then moves the end of the training set forward by `stride` time steps.
        By default, this method will return one (or a sequence of) single time series made up of
        the last point of each historical forecast.
        This time series will thus have a frequency of ``series.freq * stride``.
        If `last_points_only` is set to False, it will instead return one (or a sequence of) list of the
        historical forecasts series.
        By default, this method always re-trains the models on the entire available history, corresponding to an
        expanding window strategy. If `retrain` is set to False, the model must have been fit before. This is not
        supported by all models.
        Parameters
        ----------
        series
            The (or a sequence of) target time series used to successively train and compute the historical forecasts.
        past_covariates
            Optionally, one (or a sequence of) past-observed covariate series. This applies only if the model
            supports past covariates.
        future_covariates
            Optionally, one (or a sequence of) of future-known covariate series. This applies only if the model
            supports future covariates.
        num_samples
            Number of times a prediction is sampled from a probabilistic model. Use values `>1` only for probabilistic
            models.
        train_length
            Number of time steps in our training set (size of backtesting window to train on). Only effective when
            `retrain` is not ``False``. Default is set to `train_length=None` where it takes all available time steps
            up until prediction time, otherwise the moving window strategy is used. If larger than the number of time
            steps available, all steps up until prediction time are used, as in default case. Needs to be at least
            `min_train_series_length`.
        start
            Optionally, the first point in time at which a prediction is computed for a future time.
            This parameter supports: ``float``, ``int`` and ``pandas.Timestamp``, and ``None``.
            If a ``float``, the parameter will be treated as the proportion of the time series
            that should lie before the first prediction point.
            If an ``int``, the parameter will be treated as an integer index to the time index of
            `series` that will be used as first prediction time.
            If a ``pandas.Timestamp``, the time stamp will be used to determine the first prediction time
            directly.
            If ``None``, the first prediction time will automatically be set to:
                 - the first predictable point if `retrain` is ``False``, or `retrain` is a Callable and the first
                 predictable point is earlier than the first trainable point.
                 - the first trainable point if `retrain` is ``True`` or ``int`` (given `train_length`),
                 or `retrain` is a Callable and the first trainable point is earlier than the first predictable point.
                 - the first trainable point (given `train_length`) otherwise
            Note: Raises a ValueError if `start` yields a time outside the time index of `series`.
            Note: If `start` is outside the possible historical forecasting times, will ignore the parameter
            (default behavior with ``None``) and start at the first trainable/predictable point.
        forecast_horizon
            The forecast horizon for the predictions.
        stride
            The number of time steps between two consecutive predictions.
        retrain
            Whether and/or on which condition to retrain the model before predicting.
            This parameter supports 3 different datatypes: ``bool``, (positive) ``int``, and
            ``Callable`` (returning a ``bool``).
            In the case of ``bool``: retrain the model at each step (`True`), or never retrains the model (`False`).
            In the case of ``int``: the model is retrained every `retrain` iterations.
            In the case of ``Callable``: the model is retrained whenever callable returns `True`.
            The callable must have the following positional arguments:
                - `counter` (int): current `retrain` iteration
                - `pred_time` (pd.Timestamp or int): timestamp of forecast time (end of the training series)
                - `train_series` (TimeSeries): train series up to `pred_time`
                - `past_covariates` (TimeSeries): past_covariates series up to `pred_time`
                - `future_covariates` (TimeSeries): future_covariates series up
                  to `min(pred_time + series.freq * forecast_horizon, series.end_time())`
            Note: if any optional `*_covariates` are not passed to `historical_forecast`, ``None`` will be passed
            to the corresponding retrain function argument.
            Note: some models do require being retrained every time and do not support anything other
            than `retrain=True`.
        overlap_end
            Whether the returned forecasts can go beyond the series' end or not.
        last_points_only
            Whether to retain only the last point of each historical forecast.
            If set to True, the method returns a single ``TimeSeries`` containing the successive point forecasts.
            Otherwise, returns a list of historical ``TimeSeries`` forecasts.
        verbose
            Whether to print progress.
        show_warnings
            Whether to show warnings related to parameters `start`, and `train_length`.
        Returns
        -------
        TimeSeries or List[TimeSeries] or List[List[TimeSeries]]
            If `last_points_only` is set to True and a single series is provided in input,
            a single ``TimeSeries`` is returned, which contains the the historical forecast
            at the desired horizon.
            A ``List[TimeSeries]`` is returned if either `series` is a ``Sequence`` of ``TimeSeries``,
            or if `last_points_only` is set to False. A list of lists is returned if both conditions are met.
            In this last case, the outer list is over the series provided in the input sequence,
            and the inner lists contain the different historical forecasts.
        """
        # this is mostly ripped straight from darts.models.forecasting.forecasting_model.ForecastingModel.historical_forecasts
        # however it has be heavily reduced to be restricted to our use case

        # we will never retrain the model and have removed the functionality to do so
        assert retrain is False

        model: Ensemble = self

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

    def save(self, path: str):
        for i, contributing_model in enumerate(self.contributing_models):
            contributing_model.save(os.path.join(path, f"contributing_model_{i}"))

        self.combiner.save(os.path.join(path, "combiner"))

        models = self.contributing_models
        combiner = self.combiner

        self.contributing_models = [None] * len(self.contributing_models)
        self.combiner = combiner.__class__

        with open(os.path.join(path, "ensemble"), "wb") as f:
            pickle.dump(self, f)

        self.contributing_models = models
        self.combiner = combiner

    @staticmethod
    def load(path: str, load_cpu: bool) -> "Ensemble":
        with open(os.path.join(path, "ensemble"), "rb") as f:
            ensemble = pickle.load(f)

        for i in range(len(ensemble.contributing_models)):
            ensemble.contributing_models[i] = ContributingModel.load(
                os.path.join(path, f"contributing_model_{i}"), load_cpu
            )

        ensemble.combiner = ensemble.combiner.load(os.path.join(path, "combiner"))

        return ensemble

    def _compute_and_stack_contributing_predictions(self,
                                                    n: int,
                                                    series: TimeSeries,
                                                    past_covariates: Optional[TimeSeries] = None,
                                                    future_covariates: Optional[TimeSeries] = None) -> TimeSeries:
        predictions = [
            contributing_model.predict(n, series=series, past_covariates=past_covariates, future_covariates=future_covariates, verbose=False)
            for contributing_model in self.contributing_models
        ]

        predictions = reduce(self._stack_op, predictions)

        return predictions

    @staticmethod
    def _stack_op(a: TimeSeries, b: TimeSeries) -> TimeSeries:
        return a.stack(b)

    def _model_encoder_settings(self):
        raise NotImplementedError()
