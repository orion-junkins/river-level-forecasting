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
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
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
        -121.60385476,
        47.34862724999999,
        -121.37847998,
        47.46295243
    ],
    "properties": {
        "gauge_id": "12143400",
        "noaa_id": "GARW1",
        "gauge_name": "SF SNOQUALMIE RIVER AB ALICE CREEK NEAR GARCIA, WA",
        "area_sq_km": 107.831
    },
    "geometry": {
        "type": "MultiPoint",
        "coordinates": [
            [
                -121.7,
                47.3
            ],
            [
                -121.6,
                47.3
            ],
            [
                -121.5,
                47.3
            ],
            [
                -121.4,
                47.3
            ],
            [
                -121.3,
                47.3
            ],
            [
                -121.7,
                47.4
            ],
            [
                -121.6,
                47.4
            ],
            [
                -121.5,
                47.4
            ],
            [
                -121.4,
                47.4
            ],
            [
                -121.3,
                47.4
            ],
            [
                -121.7,
                47.5
            ],
            [
                -121.6,
                47.5
            ],
            [
                -121.5,
                47.5
            ],
            [
                -121.4,
                47.5
            ],
            [
                -121.3,
                47.5
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

forecaster = InferenceForecaster(inference_catchment_data, "trained_models/RNN/Huber/remote_50")
model = forecaster.model
all_y = forecaster.dataset.y.copy()
# %%
forecaster.predict_average()
# %%
dfs = []
#%%
for i in range(50):
    forecaster.dataset.y = forecaster.dataset.y[:-12]
    ensemble_pred = forecaster.predict()[12].pd_dataframe()
    ensemble_pred.rename(columns={"level": "ensemble"}, inplace=True)
    average_pred = forecaster.predict_average()[12].pd_dataframe()
    geo_average_pred = forecaster.predict_geo_average()[12].pd_dataframe()
    average_pred.rename(columns={"0": "average"}, inplace=True)
    geo_average_pred.rename(columns={"0": "geo_average"}, inplace=True)

    contrib_preds = forecaster.predict_contributing_models()[12].pd_dataframe()

    rename_mapping = {}
    for id, col in enumerate(contrib_preds.columns):
        rename_mapping[col] = f'contrib_{id}'

    contrib_preds.rename(columns=rename_mapping, inplace=True)

    df = ensemble_pred.join(contrib_preds)
    df = df.join(average_pred)
    df = df.join(geo_average_pred)

    dfs.append(df)


# %%

df_pred = pd.concat(dfs)

df_true = forecaster.dataset.target_scaler.inverse_transform(all_y).pd_dataframe()


df = df_pred.copy()
df["true"] = df_true["level"]
# df = df.drop(df[df['true'] < -1000].index)
df = df[::-1]

#%%


df.to_csv("test_1.csv")
# %%
# Assuming your DataFrame is named 'df'
# Selecting the columns with "Contrib" prefix
contrib_columns = [col for col in df.columns if col.startswith('contrib_')]

# Create plot objects for each line
contrib_plots = df[contrib_columns].plot(color='green', legend=False)
ensemble_plot = df['ensemble'].plot(color='blue', legend=False)
true_plot = df['true'].plot(color='red', legend=False)
average_plot = df['average'].plot(color='black', legend=False)
geo_average_plot = df['geo_average'].plot(color='black', legend=False)

# Create custom proxy artists for legend
contrib_patch = mpatches.Patch(color='green', label='Contributing Model Predictions')
ensemble_patch = mpatches.Patch(color='blue', label='Ensemble Prediction')
true_patch = mpatches.Patch(color='red', label='True Recorded')
average_patch = mpatches.Patch(color='black', label='Average')
geo_average_patch = mpatches.Patch(color='black', label='Geo Average')

# Adding the legend with custom proxy artists outside the plot
plt.legend(handles=[contrib_patch, ensemble_patch, true_patch, average_patch, geo_average_patch],
           bbox_to_anchor=(1.05, 1),
           loc='upper left')

# Displaying the plot
plt.savefig(f'output_chart_{target["properties"]["gauge_id"]}_5.png')
plt.show()
# %%
