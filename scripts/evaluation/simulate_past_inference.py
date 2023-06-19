import argparse
import pandas as pd

try:
    from rlf.aws_dispatcher import AWSDispatcher
    from rlf.forecasting.catchment_data import CatchmentData
    from rlf.forecasting.data_fetching_utilities.level_provider.level_provider_nwis import LevelProviderNWIS
    from rlf.forecasting.data_fetching_utilities.weather_provider.aws_weather_provider import AWSWeatherProvider
    from rlf.forecasting.inference_forecaster import InferenceForecaster
    from rlf.forecasting.training_helpers import get_columns, get_coordinates_for_catchment, get_recent_available_timestamps, get_level_true
except ImportError as e:
    print("Import error on rlf packages. Ensure rlf and its dependencies have been installed into the local environment.")
    print(e)
    exit(1)


def main(args: argparse.Namespace) -> int:
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

    level_true = get_level_true(timestamps, inference_level_provider, args.forecast_window)
    all_level_data = []
    all_level_data.append(level_true)
    skipped_timestamps = []

    # Run inference for each timestamp, storing the all_level_data in a List[DataFrame]
    for timestamp in timestamps:
        try:
            # Set the weather and level providers to the correct timestamps
            inference_weather_provider.set_timestamp(timestamp)
            inference_level_provider.set_timestamp(timestamp)

            # Run inference -> DataFrame
            inference_catchment_data = CatchmentData(args.gauge_id, inference_weather_provider, inference_level_provider, columns=columns)
            forecaster = InferenceForecaster(inference_catchment_data, args.trained_model_dir, load_cpu=False)
            forecast = forecaster.predict(args.forecast_window).pd_dataframe()
            forecast.rename(columns={"level": timestamp}, inplace=True)
            all_level_data.append(forecast)

        except FileNotFoundError:
            # If data is not available for the current timestamp in AWS, skip it
            # This occurs if data was not fully fetched properly
            skipped_timestamps.append(timestamp)
            continue

    # Concatenate all_level_data into a single DataFrame and save to CSV
    merged = pd.concat(all_level_data, axis=1)
    merged.to_csv(args.out_file)

    # Print skipped timestamps
    if len(skipped_timestamps) > 0:
        for timestamp in skipped_timestamps:
            print(f"Skipped timestamp {timestamp} due to missing data.")

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("gauge_id")
    parser.add_argument('-d', '--data_file', type=str, default='data/catchments_short.json', help='input file with catchment definitions, in JSON format')
    parser.add_argument('-o', '--out_file', type=str, default='out.csv', help='output file for generated forecasts, in CSV format')
    parser.add_argument('-c', '--columns_file', type=str, default='data/columns.txt', help='input text file with list of columns to use, one per line')
    parser.add_argument('-m', '--trained_model_dir', type=str, default='trained_models', help='directory containing trained_models')
    parser.add_argument('-i', '--num_inferences', type=int, default=5, help='the number of cached samples to run inference for')
    parser.add_argument('-w', '--forecast_window', type=int, default=96, help='the number of timesteps to predict at each inference')

    args = parser.parse_args()
    exit(main(args))
