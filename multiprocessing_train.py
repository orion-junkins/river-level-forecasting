from functools import partial, reduce
import multiprocessing
from multiprocessing import Pool, Process
import os
import time

from darts.models.forecasting.linear_regression_model import LinearRegressionModel
from darts.models.forecasting.regression_ensemble_model import RegressionEnsembleModel
from darts.models.forecasting.rnn_model import RNNModel
from darts.models.forecasting.random_forest import RandomForest
from darts.timeseries import TimeSeries
import pandas as pd

from rlf.aws_dispatcher import AWSDispatcher
from rlf.forecasting.catchment_data import CatchmentData
from rlf.forecasting.data_fetching_utilities.weather_provider.aws_weather_provider import AWSWeatherProvider
from rlf.forecasting.data_fetching_utilities.coordinate import Coordinate
from rlf.forecasting.data_fetching_utilities.level_provider.level_provider_nwis import LevelProviderNWIS
from rlf.forecasting.training_dataset import TrainingDataset
from rlf.forecasting.training_forecaster import TrainingForecaster
from rlf.models.contributing_model import ContributingModel
from rlf.models.ensemble import Ensemble


target = {
    "type": "Feature",
    "bbox": [-122.60881365, 44.77735762, -122.13162998, 44.89996432],
    "properties": {
        "gauge_id": "14182500",
        "gauge_name": "LITTLE NORTH SANTIAM RIVER NEAR MEHAMA, OR",
        "area_sq_km": 292.475
    },
    "geometry": {
        "type": "MultiPoint",
        "coordinates": [
            [-122.7, 44.7], [-122.6, 44.7], [-122.5, 44.7], [-122.4, 44.7], [-122.3, 44.7],
            [-122.2, 44.7], [-122.1, 44.7], [-122.7, 44.8], [-122.6, 44.8], [-122.5, 44.8],
            [-122.4, 44.8], [-122.3, 44.8], [-122.2, 44.8], [-122.1, 44.8], [-122.7, 44.9],
            [-122.6, 44.9], [-122.5, 44.9], [-122.4, 44.9], [-122.3, 44.9], [-122.2, 44.9],
            [-122.1, 44.9]
        ]
    }
}


columns_to_use = [
    "apparent_temperature",
    "cloudcover",
    "cloudcover_high",
    "cloudcover_low",
    "cloudcover_mid",
    "dewpoint_2m",
    "et0_fao_evapotranspiration",
    "precipitation",
    "pressure_msl",
    "rain",
    "relativehumidity_2m",
    "snowfall",
    "surface_pressure",
    "temperature_2m",
    "vapor_pressure_deficit",
    "winddirection_10m",
    "windgusts_10m",
    "windspeed_10m"
]


def build_custom_ensemble(dataset):
    # hidden_dim=256,50
    # n_rnn_layers=4,5
    # n_epochs=20,2
    contributing_models = [
        ContributingModel(
            RNNModel(
                120,
                n_epochs=20,
                random_state=42,
                model="GRU",
                hidden_dim=256,
                n_rnn_layers=4,
                dropout=0.01,
                training_length=144,
                force_reset=True,
                batch_size=64,
                pl_trainer_kwargs={
                    "accelerator": "gpu",
                    "enable_progress_bar": False
                }
            ),
            prefix)
        for prefix in dataset.subsets
    ]

    combiner = LinearRegressionModel(lags=None, lags_future_covariates=[0], fit_intercept=False)
    # combiner = RandomForest(lags_future_covariates=[0])

    model = Ensemble(combiner, contributing_models, 365*24*3, combiner_train_stride=5)

    return model


def train_contributing_model(
    directory: str,
    gauge_id: str,
    model_index: int,
    series,
    future_covariates,
    past_covariates,
    prediction_details: dict
):
    import logging
    logging.getLogger("pytorch_lightning.utilities.rank_zero").setLevel(logging.ERROR)
    logging.getLogger("pytorch_lightning.accelerators.cuda").setLevel(logging.ERROR)
    logging.getLogger("pytorch_lightning.callbacks.model_summary").setLevel(logging.ERROR)

    model = ContributingModel.load(os.path.join(directory, "untrained_models", gauge_id, f"contributing_model_{model_index}"))

    print(f"Begin fitting model {model_index}")
    model.fit(
        series=series,
        past_covariates=past_covariates,
        future_covariates=future_covariates)

    model.save(os.path.join(directory, "trained_models", gauge_id, f"contributing_model_{model_index}"))

    if prediction_details:
        print(f"Begin predicting model {model_index}")
        predictions = model.historical_forecasts(
            series=prediction_details["series"],
            past_covariates=past_covariates,
            future_covariates=future_covariates,
            start=prediction_details["combiner_start"],
            last_points_only=True,
            retrain=False,
            forecast_horizon=prediction_details["target_horizon"],
            stride=prediction_details["combiner_train_stride"],
            verbose=False,
            show_warnings=False
        ).pd_dataframe()
        predictions.to_parquet(os.path.join(directory, "predictions", gauge_id, f"contributing_predictions_{model_index}.parquet"))

    return model_index


def starmap_wrapper(args, f):
    return f(*args)


def main():
    # 7.71 mins for 2
    # 18.24 mins for 5

    coordinates = [Coordinate(lon, lat) for lon, lat in target["geometry"]["coordinates"]][:2]
    weather_provider = AWSWeatherProvider(coordinates, AWSDispatcher("all-weather-data", "open-meteo"))
    level_provider = LevelProviderNWIS(target["properties"]["gauge_id"])
    catchment_data = CatchmentData(target["properties"]["gauge_id"], weather_provider, level_provider, columns=columns_to_use)
    dataset = TrainingDataset(catchment_data)

    ensemble = build_custom_ensemble(dataset)

    contributing_model_y = dataset.y_train[:-ensemble._combiner_holdout_size]
    combiner_start = len(dataset.y_train) - ensemble._combiner_holdout_size + ensemble.contributing_models[0].input_chunk_length

    prediction_details = {
        "series": dataset.y_train,
        "combiner_start": combiner_start,
        "target_horizon": ensemble._target_horizon,
        "combiner_train_stride": ensemble._combiner_train_stride
    }

    jobs = [
        (
            "models",
            "14182500",
            i,
            contributing_model_y,
            dataset.X_train,
            None,
            prediction_details
        ) for i in range(len(ensemble.contributing_models))
    ]

    os.makedirs("models/untrained_models/14182500", exist_ok=True)
    os.makedirs("models/trained_models/14182500", exist_ok=True)
    os.makedirs("models/predictions/14182500", exist_ok=True)

    for i, contributing_model in enumerate(ensemble.contributing_models):
        contributing_model.save(f"models/untrained_models/14182500/contributing_model_{i}")

    multiprocessing.set_start_method("spawn", force=True)

    start_time = time.perf_counter()

    with Pool(5, maxtasksperchild=1) as p:
        for i in p.imap_unordered(partial(starmap_wrapper, f=train_contributing_model), jobs):
            print(f"Completed model {i}")

    end_time = time.perf_counter()
    print(f"**** Total time: {(end_time - start_time) / 60:.2f} m")

    predictions = [
        TimeSeries.from_dataframe(
            pd.read_parquet(f"models/predictions/14182500/contributing_predictions_{i}.parquet")
        )
        for i in range(len(ensemble.contributing_models))
    ]

    predictions = reduce(Ensemble._stack_op, predictions)

    ensemble.combiner.fit(series=dataset.y_train.slice_intersect(predictions), future_covariates=predictions)


if __name__ == "__main__":
    main()

