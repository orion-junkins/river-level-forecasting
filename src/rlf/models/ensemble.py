from functools import reduce
from typing import List

from darts.timeseries import TimeSeries
from darts.models.forecasting.forecasting_model import GlobalForecastingModel
from rlf.models.contributing_model import ContributingModel


class Ensemble:

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
        self.combiner = combiner
        self.contributing_models = contributing_models
        self._combiner_holdout_size = combiner_holdout_size
        self._target_horizon = target_horizon
        self._combiner_train_stride = combiner_train_stride

    def fit(self, series: TimeSeries, *, past_covariates: TimeSeries = None, future_covariates: TimeSeries = None) -> "Ensemble":
        contributing_model_y = series[:-self._combiner_holdout_size]

        combiner_start = len(series) - self._combiner_holdout_size + self.contributing_models[0].input_chunk_length

        predictions: List[TimeSeries] = []

        for contributing_model in self.contributing_models:
            contributing_model.fit(series=contributing_model_y, past_covariates=past_covariates, future_covariates=future_covariates)
            predictions.append(contributing_model.historical_forecasts(
                series=series,
                past_covariates=past_covariates,
                future_covariates=future_covariates,
                start=combiner_start,
                last_points_only=True,
                retrain=False,
                forecast_horizon=self._target_horizon,
                stride=self._combiner_train_stride,
            ))

        del contributing_model_y

        predictions = reduce(self._stack_op, predictions)
        self.combiner.fit(series=series.slice_intersect(predictions), future_covariates=predictions)

        del predictions

        for contributing_model in self.contributing_models:
            contributing_model.fit(series=series, past_covariates=past_covariates, future_covariates=future_covariates)

        return self

    def predict(self, n: int, *, series: TimeSeries, past_covariates: TimeSeries = None, future_covariates: TimeSeries = None) -> TimeSeries:
        predictions = [
            contributing_model.predict(n, series=series, past_covariates=past_covariates, future_covariates=future_covariates)
            for contributing_model in self.contributing_models
        ]

        predictions = reduce(self._stack_op, predictions)

        return self.combiner.predict(n, series=series, future_covariates=predictions)

    def historical_forecasts(
            self,
            *args,
            **kwargs) -> TimeSeries:
        # these arguments are a bit lazy but it gets the job done
        predictions: List[TimeSeries] = [
            contributing_model.historical_forecasts(*args, **kwargs)
            for contributing_model in self.contributing_models
        ]

        predictions = reduce(self._stack_op, predictions)

        kwargs["future_covariates"] = predictions

        return self.combiner.historical_forecasts(*args, **kwargs)

    def backtest(self, *args, **kwargs) -> float:
        """See GlobalForecastingModel.backtest for documentation on parameters.

        Returns:
            float: Error score for the model.
        """
        return GlobalForecastingModel.backtest(self, *args, **kwargs)

    @staticmethod
    def _stack_op(a: TimeSeries, b: TimeSeries) -> TimeSeries:
        return a.stack(b)
