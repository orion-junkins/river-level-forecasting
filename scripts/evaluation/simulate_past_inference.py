import argparse
import json
import pandas as pd
from typing import List, Optional

from rlf.aws_dispatcher import AWSDispatcher

try:
    from rlf.forecasting.catchment_data import CatchmentData
    from rlf.forecasting.data_fetching_utilities.coordinate import Coordinate
    from rlf.forecasting.data_fetching_utilities.level_provider.level_provider_nwis import LevelProviderNWIS
    from rlf.forecasting.data_fetching_utilities.weather_provider.aws_weather_provider import AWSWeatherProvider
    from rlf.forecasting.inference_forecaster import InferenceForecaster
except ImportError as e:
    print("Import error on rlf packages. Ensure rlf and its dependencies have been installed into the local environment.")
    print(e)
    exit(1)


def get_columns(column_file: str) -> List[str]:
    """Get the list of columns from a text file.

    Args:
        column_file (str): path to the text file containing the columns.

    Returns:
        List[str]: List of columns to use
    """
    with open(column_file) as f:
        return [c.strip() for c in f.readlines()]


def get_coordinates_for_catchment(filename: str, gauge_id: str) -> Optional[List[Coordinate]]:
    """Get the list of coordinates for a specific gauge ID from a geojson file.

    Args:
        filename (str): geojson file that contains catchment information.
        gauge_id (str): gauge ID to retrieve coordinates for.

    Returns:
        Optional[List[Coordinate]]: List of coordinates for the given gauge or None if the gauge could not be found.
    """
    with open(filename) as f:
        target = json.load(f)

    for feature in target["features"]:
        if feature["properties"]["gauge_id"] == gauge_id:
            coordinates = [Coordinate(lon, lat) for lon, lat in feature["geometry"]["coordinates"]]
            return coordinates

    return None


def get_recent_available_timestamps(aws_dispatcher: AWSDispatcher, num_timestamps: int) -> List[str]:
    """Get a list of recent timestamps available in AWS.

    Args:
        aws_dispatcher (AWSDispatcher): AWSDispatcher to use to list files.
        num_timestamps (int): Number of timestamps to fetch.

    Returns:
        List[str]: List of timestamps available in AWS.
    """
    files = aws_dispatcher.list_files("current")
    timestamps = list(map(lambda x: x.split("/")[-1], files))
    timestamps = timestamps[-num_timestamps:]
    return timestamps


def main(args: List[str]) -> int:
    # Fetch columns list from specified data file
    columns = get_columns(args.columns_file)

    # Fetch coordinates for the specified gauge ID
    coordinates = get_coordinates_for_catchment(args.data_file, args.gauge_id)
    if coordinates is None:
        print(f"Unable to locate {args.gauge_id} in catchment data file.")
        return 1

    # Create AWSDispatcher and load available timestamps
    aws_dispatcher = AWSDispatcher("all-weather-data", "open-meteo")
    timestamps = get_recent_available_timestamps(aws_dispatcher, args.num_inferences)

    # Ceate weather and level providers for inference
    inference_weather_provider = AWSWeatherProvider(coordinates, aws_dispatcher=aws_dispatcher)
    inference_level_provider = LevelProviderNWIS(args.gauge_id)

    # Run inference for each timestamp, storing the predictions in a List[DataFrame]
    predictions = []
    for timestamp in timestamps:
        try:
            # Set the weather and level providers to the correct timestamps
            inference_weather_provider.set_timestamp(timestamp)
            inference_level_provider.set_timestamp(timestamp)

            # Run inference -> DataFrame
            inference_catchment_data = CatchmentData(args.gauge_id, inference_weather_provider, inference_level_provider, columns=columns)
            forecaster = InferenceForecaster(inference_catchment_data, args.trained_model_dir, load_cpu=True)
            forecast = forecaster.predict(args.forecast_window).pd_dataframe()
            forecast.rename(columns={"level": timestamp}, inplace=True)
            predictions.append(forecast)

        except FileNotFoundError:
            # If data is not available for the current timestamp in AWS, skip it
            # This occurs if data was not fully fetched properly
            print(f"Data not available for timestamp {timestamp}. Skipping.")
            continue

    # Concatenate the predictions into a single DataFrame and save to CSV
    merged = pd.concat(predictions, axis=1)
    merged.to_csv(args.out_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser() 
    parser.add_argument("gauge_id")
    parser.add_argument('-d', '--data_file', nargs=1, type=str, default='data/catchments_short.json', help='input file with catchment definitions, in JSON format')
    parser.add_argument('-o', '--out_file', nargs=1, type=str, default='out.csv', help='output file for generated forecasts, in CSV format')
    parser.add_argument('-c', '--columns_file', nargs=1, type=str, default='data/columns.txt', help='input text file with list of columns to use, one per line')
    parser.add_argument('-m', '--trained_model_dir', nargs=1, type=str, default='trained_models', help='directory containing trained_models')
    parser.add_argument('-s', '--num_inferences', nargs=1, type=int, default=5, help='=the number of cached sampled to run inference for')
    parser.add_argument('-w', '--forecast_window', nargs=1, type=int, default=96, help='the nummber of timesteps to predict at each inference')

    args = parser.parse_args()
    exit(main(args))
