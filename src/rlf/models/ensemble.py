from functools import reduce

class Ensemble:

    def __init__(
            self,
            combiner,
            contributing_models,
            combiner_holdout_size: int,
            target_horizon: int = 6,
            combiner_train_stride: int = 25,
        ) -> None:
        self.combiner = combiner
        self.contributing_models = contributing_models
        self._combiner_holdout_size = combiner_holdout_size
        self._target_horizon = target_horizon
        self._combiner_train_stride = combiner_train_stride

    def fit(self, series, future_covariates) -> "Ensemble":
        contributing_model_y = series[:-self._combiner_holdout_size]

        combiner_start = len(series) - self._combiner_holdout_size + self.contributing_models[0].input_chunk_length

        predictions = []

        for contributing_model in self.contributing_models:
            contributing_model.fit(contributing_model_y, future_covariates=future_covariates)
            predictions.append(contributing_model.historical_forecasts(
                series=series,
                future_covariates=future_covariates,
                start=combiner_start,
                last_points_only=True,
                retrain=False,
                forecast_horizon=self._target_horizon,
                stride=self._combiner_train_stride,
            ))

        predictions = reduce(lambda a, b: a.stack(b), predictions)
        self.combiner.fit(series.slice_intersect(predictions), future_covariates=predictions)

        for contributing_model in self.contributing_models:
            contributing_model.fit(series, future_covariates=future_covariates)

        return self
