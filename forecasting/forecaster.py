import sys
import pandas as pd
from forecasting.dataset import Dataset
from forecasting.general_utilities.logging_utils import build_logger
from sklearn.linear_model import LinearRegression
from collections import defaultdict
import numpy as np

class Forecaster:
    """
    Highest level user facing class. Each instance is specific to a single gauge site/catchment.
    Managers training data, inference data, model fitting and inference of trained model.
    Allows the user to query for specific forecasts or forecast ranges.
    """
    def __init__(self, catchment_data, catchment_models, test_size=0.1, validation_size=0.2) -> None:
        """
        Fetches data from provided forecast site and generates processed training and inference sets.
        Builds the specified model.
        """
        self.logger = build_logger(log_level='INFO')
        self.name = catchment_data.name
        self.dataset = Dataset(catchment_data, test_size=test_size, validation_size=validation_size)

        self.catchment_models = catchment_models
        self.ensemble_model = LinearRegression()
        
        self.historical_trib_forecasts = None
        self.built_historical_forecasts = defaultdict(None)


    def fit(self, epochs=1):
        # Fit catchment_models on Xs
        self.fit_catchment_models(epochs=epochs)

        # Predict validation set
        historical_trib_forecasts = self.historical_catchment_forecasts()
        historical_trib_forecasts.dropna(inplace=True)
        self.historical_trib_forecasts = historical_trib_forecasts

        # Fit ensemble model on predictions
        self.fit_ensemble_model(historical_trib_forecasts)


    def fit_catchment_models(self, epochs=10):
        Xs = self.dataset.Xs_train
        y = self.dataset.y_train
        for index, (X, model) in enumerate(zip(Xs, self.catchment_models)):
            print(f"Training Model {index} for {epochs} epochs")
            model.fit(series=y, past_covariates=X, epochs=epochs)
    

    def fit_ensemble_model(self, df):
        """
        Given a dataframe with columns level_0, level_1...level_11 and level_true,
        train self.ensemble model to predict level_true based on other cols
        """
        y = df['level_true']
        X = df.drop(columns=['level_true'])
        self.ensemble_model.fit(X, y)


    def predict(self, num_timesteps=6):
        # Generate catchment model predictions
        preds = self.predict_catchment_models(num_timesteps=num_timesteps)

        # Generate ensembled prediction
        y_ensembled = self.ensemble_model.predict(preds)

        preds['level_pred'] = y_ensembled
        return preds


    def predict_catchment_models(self, num_timesteps, num_samples=1):
        """
        """
        y = self.dataset.y_current
        Xs = self.dataset.Xs_current

        models = self.catchment_models
        target_scaler = self.dataset.target_scaler
        y_preds = []

        for index, (X, model) in enumerate(zip(Xs, models)):
            y_pred = model.predict(n=num_timesteps, series=y, past_covariates=X, num_samples=num_samples)
            y_pred = target_scaler.inverse_transform(y_pred)
            y_pred = y_pred.pd_dataframe()
            y_pred.rename(columns={"level": "level_"+str(index)}, inplace=True)
            y_preds.append(y_pred)

        df_y_preds = pd.concat(y_preds, axis=1)

        return df_y_preds


    def get_forecast(self, num_timesteps=24, update_dataset=True):
        if update_dataset:
            self.update_dataset()
            
        # Generate forecast
        y_preds = self.predict(num_timesteps)
        
        # Grab recent true data
        y_recent = self.get_y_df(data_partition="current")

        # Join all data into single dataframe
        frames = [y_recent, y_preds]
        df = pd.concat(frames, axis=1)

        return df


    def historical_catchment_forecasts(self, data_partition="validation", **kwargs):
        y = self.get_y(data_partition)
        Xs = self.get_Xs(data_partition)

        models = self.catchment_models
        target_scaler = self.dataset.target_scaler
        y_preds = []

        for index, (X, model) in enumerate(zip(Xs, models)):
            print(f"Generating historical forecasts for model {index}")
            y_pred = model.historical_forecasts(series=y, past_covariates=X, start=0.02, retrain=False, last_points_only=True, verbose=True , **kwargs)
            y_pred = target_scaler.inverse_transform(y_pred)
            y_pred = y_pred.pd_dataframe()
            y_pred = self.rename_pred_cols(y_pred, index)
            y_preds.append(y_pred)

        df_y_preds = pd.concat(y_preds, axis=1)
        y_true = target_scaler.inverse_transform(y).pd_dataframe()
        df_y_preds['level_true'] = y_true['level']

        return df_y_preds

    def historical_forecasts(self, data_partition="test", **kwargs):
        if self.historical_forecasts[data_partition] != None:
            return self.historical_forecasts[data_partition]

        historical_forecasts = self.historical_catchment_forecasts(data_partition=data_partition, **kwargs)
        X = historical_forecasts.drop(columns=['level_true'])
        y_ensembled = self.ensemble_model.predict(X)
        historical_forecasts['level_pred'] = y_ensembled

        self.built_historical_forecasts[data_partition] = historical_forecasts

        return historical_forecasts


    def score(self, data_partition = "test"):
        if self.built_historical_forecasts[data_partition] == None:
            self.historical_forecasts(data_partition=data_partition)

        y_all = self.historical_forecasts[data_partition]
        y_true = y_all['level_true']
        y_hats = y_all.drop(columns=['level_true'])
        
        mae_df = pd.DataFrame()
        mape_df = pd.DataFrame()

        for col in y_hats.columns:
            y_hat = y_hats[col]
            ensembled_mae = self.mae(y_true, y_hat)
            mae_df[col] = ensembled_mae
            ensembled_mape = self.mape(y_true, y_hat)
            mape_df[col] = ensembled_mape
        
        return (mae_df, mape_df)


    def mae(self, y, y_hat):
        return np.mean(np.abs(y - y_hat))


    def mape(self, y, y_hat):
        return np.mean(np.abs((y - y_hat)/y)*100)


    # Utilities & Helpers
    def update_dataset(self) -> None:
        """
        Force an update to ensure inference data is up to date. Run at least hourly when forecasting.
        """
        self.dataset.update()


    def get_y(self, data_partition):
        if data_partition == "current":
            y = self.dataset.y_current
        elif data_partition == "train":
            y = self.dataset.y_train
        elif data_partition == "validation":
            y = self.dataset.y_validation
        elif data_partition == "test":
            y = self.dataset.y_test
        else:
            self.logger.error("Invalid argument")
            sys.exit()
        return y
    

    def get_y_df(self, data_partition):
        y = self.get_y(data_partition)
        target_scaler = self.dataset.target_scaler
        y = target_scaler.inverse_transform(y)
        df_y = y.pd_dataframe()
        df_y.rename(columns={"level": "level_true"}, inplace=True)
        return df_y
    

    def get_Xs(self, data_partition):
        if data_partition == "current":
            Xs = self.dataset.Xs_current
        elif data_partition == "train":
            Xs = self.dataset.Xs_train
        elif data_partition == "validation":
            Xs = self.dataset.Xs_validation
        elif data_partition == "test":
            Xs = self.dataset.Xs_test
        else:
            self.logger.error("Invalid argument")
            sys.exit()
        
        return Xs

    def rename_pred_cols(self, y_pred, index):
        if "level" in y_pred.columns:
            y_pred.rename(columns={"level": "level_"+str(index)}, inplace=True)
        elif "0" in y_pred.columns:
            y_pred.rename(columns={"0": "level_"+str(index)}, inplace=True)