from datetime import datetime
import json
import math
import os
import re
import requests

from darts.timeseries import TimeSeries
import pandas as pd
import s3fs

from rlf.forecasting.data_fetching_utilities.coordinate import Coordinate
from rlf.forecasting.data_fetching_utilities.level_provider.level_provider_nwis import LevelProviderNWIS
from rlf.forecasting.data_fetching_utilities.weather_provider.api_weather_provider import APIWeatherProvider
from rlf.forecasting.catchment_data import CatchmentData
from rlf.forecasting.inference_dataset import InferenceDataset
from rlf.forecasting.inference_forecaster import InferenceForecaster


s3 = s3fs.S3FileSystem(anon=False)
s3_bucket = "s3://model-forecasts"

flow_pattern = re.compile("(\d+\.?\d*)(k?cfs)")


def parse_flow(x: str) -> float:
  value, suffix = re.match(flow_pattern, x).groups()
  value = float(value)
  if suffix == "kcfs":
    value *= 1000
  return value


def parse_datetime_from_noaa(x: str) -> datetime:
  basic_datetime = datetime.strptime(x, "%m/%d %H:%M")

  now = datetime.now()

  # since noaa predictions only go a few days out, assume any months greater than
  # the current month + 1 is actually last year
  if basic_datetime.month > now.month + 1:
    return basic_datetime.replace(year=now.year + 1)
  else:
    return basic_datetime.replace(year=now.year)


def get_noaa_predictions(target_gauge: str) -> pd.DataFrame:
    response = requests.request("GET", f"https://water.weather.gov/ahps2/hydrograph_to_xml.php?gage={target_gauge}&output=tabular")

    begin = "<!--- start forecast table --->"
    end = "<!--- end forecast table --->"

    decoded_content = response.content.decode("UTF-8")

    begin_index = decoded_content.find(begin)
    end_index = decoded_content.find(end)

    df = pd.read_html(decoded_content[begin_index + len(begin):end_index], skiprows=2)[0]
    df.columns = ["datetime", "level", "flow"]
    df["datetime"] = df["datetime"].apply(parse_datetime_from_noaa)
    df["flow"] = df["flow"].apply(parse_flow)
    df = df.set_index("datetime")

    # some NOAA predictions are every six hour
    # expand to hourly and interpolate between so that there
    # are not a bunch of nans when paired with the model data
    df = df.asfreq("H")
    df.interpolate(inplace=True)

    return df


def build_metadata_dict(target: dict) -> dict:
    metadata_dict = {
        "id": target["properties"]["gauge_id"],
        "name": target["properties"]["gauge_name"],
        "noaa_id": target["properties"].get("noaa_id", None)
    }

    return metadata_dict


def build_data_dict(dataset: InferenceDataset, predictions: TimeSeries, target: dict) -> dict:
    true_data_ts = dataset.target_scaler.inverse_transform(dataset.y)[-(7*24):]

    data_df = pd.DataFrame(
        index=pd.date_range(true_data_ts.time_index.min(), predictions.time_index.max(), freq="H")
    )
    data_df["level_true"] = true_data_ts.pd_dataframe()["level"]
    data_df["pred_model_1"] = predictions.pd_dataframe()["level"]

    if "noaa_id" in target["properties"]:
       try:
          data_df["pred_noaa"] = get_noaa_predictions(target["properties"]["noaa_id"])["flow"]
       except Exception as e:
          print("Unable to get noaa predictions:")
          print(e)

    data_dict = {}
    data_dict["timestamps"] = list(map(lambda x: x.strftime("%Y-%m-%d %H:%M:%S"), data_df.index))
    data_dict["level_true"] = list(map(lambda x: None if math.isnan(x) else x, data_df["level_true"]))
    data_dict["pred_model_1"] = list(map(lambda x: None if math.isnan(x) else x, data_df["pred_model_1"]))

    if "pred_noaa" in data_df.columns:
        data_dict["pred_noaa"] = list(map(lambda x: None if math.isnan(x) else x, data_df["pred_noaa"]))

    return data_dict


def build_json_result(dataset: InferenceDataset, predictions: TimeSeries, target: dict) -> str:
    metadata_dict = build_metadata_dict(target)
    data_dict = build_data_dict(dataset, predictions, target)

    complete_dict = {
       "metadata": metadata_dict,
       "data": data_dict
    }

    return json.dumps(complete_dict)


def run_predictions_for_target(target: dict):
    coordinates = [Coordinate(lon, lat) for lon, lat in target["geometry"]["coordinates"]]

    inference_weather_provider = APIWeatherProvider(coordinates)
    inference_level_provider = LevelProviderNWIS(target["properties"]["gauge_id"])
    inference_catchment_data = CatchmentData(target["properties"]["gauge_id"], inference_weather_provider, inference_level_provider)

    forecaster = InferenceForecaster(inference_catchment_data, "trained_models", load_cpu=True)
    predictions = forecaster.predict(96)

    json_result = build_json_result(forecaster.dataset, predictions, target)

    s3.write_text(f"{s3_bucket}/{target['properties']['gauge_id']}.json", json_result)


def handler(event, context):
    # load catchments dict
    with open("data/catchments_short.json") as f:
        catchments = json.load(f)

    for feature in catchments["features"]:
        if os.path.exists(f"trained_models/{feature['properties']['gauge_id']}"):
           try:
              run_predictions_for_target(feature)
           except Exception as e:
              print(f"Unable to run predictions for {feature['properties']['gauge_id']}")
              raise

    return


if __name__ == "__main__":
    handler(None, None)
