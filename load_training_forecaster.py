# %%
"""
# Inference Forecaster Demo

Executing inference requires loading the appropriate data, a trained model, and then making predictions.
We have bundled that functionality into only a few steps

## Data Fetching

The inference forecaster needs a `CatchmentData` object in order to access the data needed for predictions.
Building that requires a couple separate objects, but most importantly you will need to have selected a gauge ID and have the list of coordinates for that catchment.
Note: the coordinates must be the same that were trained on.
"""

# %%
import pandas as pd
import darts
from rlf.aws_dispatcher import AWSDispatcher
from rlf.forecasting.catchment_data import CatchmentData
from rlf.forecasting.data_fetching_utilities.coordinate import Coordinate
from rlf.forecasting.data_fetching_utilities.level_provider.level_provider_nwis import LevelProviderNWIS
from rlf.forecasting.data_fetching_utilities.weather_provider.api_weather_provider import APIWeatherProvider
from rlf.forecasting.data_fetching_utilities.weather_provider.aws_weather_provider import AWSWeatherProvider
from rlf.forecasting.inference_forecaster import InferenceForecaster
from rlf.forecasting.training_dataset import TrainingDataset
from rlf.forecasting.training_forecaster import TrainingForecaster
from rlf.types import GeoJSONFeature

# %%
target: GeoJSONFeature = {
    "type": "Feature",
            "bbox": [
                -121.13349597,
                47.46497187,
                -120.71424355,
                47.72382705
            ],
    "properties": {
                "gauge_id": "12458000",
                "gauge_name": "ICICLE CREEK ABOVE SNOW CREEK NEAR LEAVENWORTH, WA",
                "area_sq_km": 499.975
            },
    "geometry": {
                "type": "MultiPoint",
                "coordinates": [
                    [
                        -121.2,
                        47.4
                    ],
                    [
                        -121.1,
                        47.4
                    ],
                    [
                        -121.0,
                        47.4
                    ],
                    [
                        -120.9,
                        47.4
                    ],
                    [
                        -120.8,
                        47.4
                    ],
                    [
                        -120.7,
                        47.4
                    ],
                    [
                        -121.2,
                        47.5
                    ],
                    [
                        -121.1,
                        47.5
                    ],
                    [
                        -121.0,
                        47.5
                    ],
                    [
                        -120.9,
                        47.5
                    ],
                    [
                        -120.8,
                        47.5
                    ],
                    [
                        -120.7,
                        47.5
                    ],
                    [
                        -121.2,
                        47.6
                    ],
                    [
                        -121.1,
                        47.6
                    ],
                    [
                        -121.0,
                        47.6
                    ],
                    [
                        -120.9,
                        47.6
                    ],
                    [
                        -120.8,
                        47.6
                    ],
                    [
                        -120.7,
                        47.6
                    ],
                    [
                        -121.2,
                        47.7
                    ],
                    [
                        -121.1,
                        47.7
                    ],
                    [
                        -121.0,
                        47.7
                    ],
                    [
                        -120.9,
                        47.7
                    ],
                    [
                        -120.8,
                        47.7
                    ],
                    [
                        -120.7,
                        47.7
                    ],
                    [
                        -121.2,
                        47.8
                    ],
                    [
                        -121.1,
                        47.8
                    ],
                    [
                        -121.0,
                        47.8
                    ],
                    [
                        -120.9,
                        47.8
                    ],
                    [
                        -120.8,
                        47.8
                    ],
                    [
                        -120.7,
                        47.8
                    ]
                ]
            }
}
columns = [
    "temperature_2m",
    "relativehumidity_2m",
    "dewpoint_2m",
    "apparent_temperature",
    "pressure_msl",
    "surface_pressure",
    "precipitation",
    "snowfall",
    "cloudcover",
    "cloudcover_low",
    "cloudcover_mid",
    "cloudcover_high",
    "windspeed_10m",
    "et0_fao_evapotranspiration",
    "vapor_pressure_deficit",
    "soil_temperature_level_1",
    "soil_temperature_level_2",
    "soil_temperature_level_3",
    "soil_temperature_level_4",
    "soil_moisture_level_1",
    "soil_moisture_level_2",
    "soil_moisture_level_3",
    "soil_moisture_level_4"
]

# %%

coordinates = [Coordinate(lon, lat) for lon, lat in target["geometry"]["coordinates"]]

inference_weather_provider = APIWeatherProvider(coordinates)
inference_level_provider = LevelProviderNWIS(target["properties"]["gauge_id"])
inference_catchment_data = CatchmentData(target["properties"]["gauge_id"], inference_weather_provider, inference_level_provider, columns=columns)

forecaster = InferenceForecaster(inference_catchment_data, "trained_models")
model = forecaster.model
all_y = forecaster.dataset.y.copy()

# %%

training_weather_provider = AWSWeatherProvider(
    coordinates,
    AWSDispatcher("all-weather-data", "open-meteo")
)
training_level_provider = LevelProviderNWIS(target["properties"]["gauge_id"])
training_catchment_data = CatchmentData(
    target["properties"]["gauge_id"],
    training_weather_provider,
    training_level_provider,
    columns=columns
)


td = TrainingDataset(catchment_data=training_catchment_data)

# %%
