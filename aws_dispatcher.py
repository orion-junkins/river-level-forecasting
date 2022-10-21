import os
import pickle

import boto3

DEFAULT_LOCAL_PATH = os.path.join("data", "aws_dispatch")
os.makedirs(DEFAULT_LOCAL_PATH, exist_ok=True)


class AWSDispatcher():
    def __init__(self, river_gauge_name, model_name) -> None:
        self.river_gauge_name = river_gauge_name
        self.model_name = model_name
        self.forecaster = self.load_forecaster()

    def load_forecaster(self):
        trained_model_dir = "trained_models"
        frcstr_file = os.path.join(trained_model_dir, self.river_gauge_name, self.model_name, self.model_name + "_frcstr.pickle")
        # Load trained forecaster
        pickle_in = open(frcstr_file, "rb")
        frcstr = pickle.load(pickle_in)
        return frcstr

    def pickle_to_aws(self, payload, filename, reg_model_name, local_dir=DEFAULT_LOCAL_PATH):
        """
        Pickle the given payload locally, and upload the file to AWS

        Args:
            payload (object): any python object, typically a DataFrame
            filename (string): desired name for the file. Determines local/s3 file name.
            local_dir (path, optional): Root dir for local storage. Defaults to DEFAULT_LOCAL_PATH.
        """
        out_dir = os.path.join(local_dir, self.river_gauge_name, self.model_name)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, filename + ".pickle")
        pickle_out = open(out_path, "wb")
        pickle.dump(payload, pickle_out)
        pickle_out.close()
        s3 = boto3.client('s3')
        s3.upload_file(out_path, 'generated-forecasts', f'{self.river_gauge_name}/{self.model_name}/{reg_model_name}/{filename}.pickle')

    def rebuild_current_forecast(self, horizon=96, update_dataset=True):
        reg_model_names = self.forecaster.regression_models.keys()
        for reg_model_name in reg_model_names:
            fcast = self.forecaster.get_forecast(num_timesteps=horizon, reg_model_name=reg_model_name, update_dataset=update_dataset)

            self.pickle_to_aws(fcast, reg_model_name=reg_model_name, filename="current_forecast")

    def rebuild_historical_forecast(self, horizon=24, stride=1):
        reg_model_names = self.forecaster.regression_models.keys()
        for reg_model_name in reg_model_names:
            self.forecaster.build_historical_reg_forecasts(reg_model_name=reg_model_name, forecast_horizon=horizon, stride=stride)
            fcast = self.forecaster.historical_reg_forecasts[reg_model_name]
            self.pickle_to_aws(fcast, filename=f"historical_forecast_h{horizon}_s{stride}", reg_model_name=reg_model_name)
