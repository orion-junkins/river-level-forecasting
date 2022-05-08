import sys
import pandas as pd
import numpy as np
import pandas as pd

from forecasting.dataset import Dataset
from forecasting.general_utilities.logging_utils import build_logger
from sklearn.linear_model import LinearRegression

class Forecaster:
    """
    Highest level user facing class. Each instance is specific to a single gauge site/tributary.
    Managers training data, inference data, model fitting and inference of trained model.
    Allows the user to query for specific forecasts or forecast ranges.
    """
    def __init__(self, tributary_data, tributary_models, test_size=0.01, validation_size=0.01) -> None:
        """
        Fetches data from provided forecast site and generates processed training and inference sets.
        Builds the specified model.
        """
        self.logger = build_logger(log_level='INFO')
        self.name = tributary_data.name
        self.dataset = Dataset(tributary_data, test_size=test_size, validation_size=validation_size)

        self.tributary_models = tributary_models
        self.regression_models = {}
        
        self.historical_trib_forecasts = {}
        self.historical_reg_forecasts = {}



    # MODEL FITTING
    def fit(self, reg_model=LinearRegression(), epochs=10):
        # Check if historical forecasts have been built; build them if not
        if 'validation' not in self.historical_trib_forecasts.keys():
            print("Building historical trib forecasts")
            # Fit tributary_models on Xs
            self.fit_tributary_models(epochs=epochs)

            # Predict validation set
            self.build_historical_tributary_forecasts()

        # Fit ensemble model given built historical forecasts
        self.fit_ensemble_model(reg_model)


    def fit_tributary_models(self, epochs=10):
        """ Fit tribuatry models on the training set
        """
        Xs = self.dataset.Xs_train
        y = self.dataset.y_train

        # For every X set and model, fit
        for index, (X, model) in enumerate(zip(Xs, self.tributary_models)):
            print(f"Training Model {index} for {epochs} epochs")
            model.fit(series=y, past_covariates=X, epochs=epochs)
    

    def fit_ensemble_model(self, reg_model=LinearRegression()):
        """
        Given a dataframe with columns level_0, level_1...level_11 and level_true,
        train self.ensemble model to predict level_true based on other cols
        """
        df = self.historical_trib_forecasts['validation']
        reg_model_name = type(reg_model).__name__
        if reg_model_name in self.regression_models:
            print(f"{reg_model_name} model has already been fit!")
        else:
            print(f"Fitting regression model: {reg_model_name}")
            y = df['level_true']
            X = df.drop(columns=['level_true'])
            reg_model.fit(X, y)
            self.regression_models[reg_model_name] = reg_model


    # FUTURE FORECASTING 
    def get_forecast(self, num_timesteps=24, reg_model_name="LinearRegression", update_dataset=True):
        if update_dataset:
            self.update_dataset()
        
        # Generate forecast
        y_preds = self.predict(num_timesteps=num_timesteps, reg_model_name=reg_model_name)
        
        # Grab recent true data
        y_recent = self.get_y_df(data_partition="current")

        # Join all data into single dataframe
        frames = [y_recent, y_preds]
        df = pd.concat(frames, axis=1)

        return df


    def predict(self, num_timesteps=6, reg_model_name='LinearRegression'):
        # Generate tributary model predictions
        preds = self.predict_tributary_models(num_timesteps=num_timesteps)

        # Generate ensembled prediction
        reg_model = self.regression_models[reg_model_name]
        y_ensembled = reg_model.predict(preds)

        preds['level_pred'] = y_ensembled
        return preds


    def predict_tributary_models(self, num_timesteps, num_samples=1):
        """
        """
        y = self.dataset.y_current
        Xs = self.dataset.Xs_current

        models = self.tributary_models
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



    # HISTORICAL FORECASTING
    def build_historical_tributary_forecasts(self, data_partition="validation", **kwargs):
        y = self.get_y(data_partition)
        Xs = self.get_Xs(data_partition)

        models = self.tributary_models
        target_scaler = self.dataset.target_scaler
        y_preds = []

        for index, (X, model) in enumerate(zip(Xs, models)):
            print(f"Generating historical forecasts for model {index}")
            y_pred = model.historical_forecasts(series=y, past_covariates=X, start=0.99, retrain=False, last_points_only=True, verbose=True , **kwargs)
            y_pred = target_scaler.inverse_transform(y_pred)
            y_pred = y_pred.pd_dataframe()
            y_pred = self.rename_pred_cols(y_pred, index)
            y_preds.append(y_pred)

        df_y_preds = pd.concat(y_preds, axis=1)
        y_true = target_scaler.inverse_transform(y).pd_dataframe()
        df_y_preds['level_true'] = y_true['level']

        df_y_preds.dropna(inplace=True)
        self.historical_trib_forecasts[data_partition] = df_y_preds


    def build_historical_reg_forecasts(self, data_partition="test", reg_model_name='LinearRegression', **kwargs):
        if reg_model_name in self.historical_reg_forecasts.keys():
            return

        if reg_model_name not in self.regression_models:
            print("The specified regression model does not exist. Pass an instance to fit or check that you are specifying the correct name")
            sys.exit(2)
        
        if data_partition not in self.historical_trib_forecasts.keys():
            self.build_historical_tributary_forecasts(data_partition=data_partition, **kwargs)

        historical_forecasts = self.historical_trib_forecasts[data_partition].copy()
        X = historical_forecasts.drop(columns=['level_true'])

        reg_model = self.regression_models[reg_model_name]
        y_ensembled = reg_model.predict(X)
        historical_forecasts['level_pred'] = y_ensembled

        self.historical_reg_forecasts[reg_model_name] = historical_forecasts



    # MODEL EVALUATION
    def score(self, reg_model_name='LinearRegression'):
        if reg_model_name not in self.historical_reg_forecasts.keys():
            self.build_historical_reg_forecasts(reg_model_name=reg_model_name)

        y_all = self.historical_reg_forecasts[reg_model_name]
        y_true = y_all['level_true']
        y_hats = y_all.drop(columns=['level_true'])
        
        mae_scores = {}
        mape_scores = {}

        for col in y_hats.columns:
            y_hat = y_hats[col]
            ensembled_mae = self.mae(y_true, y_hat)
            mae_scores[col] = ensembled_mae
            ensembled_mape = self.mape(y_true, y_hat)
            mape_scores[col] = ensembled_mape
        
        return (mae_scores, mape_scores)


    def mae(self, y, y_hat):
        return np.mean(np.abs(y - y_hat))


    def mape(self, y, y_hat):
        return np.mean(np.abs((y - y_hat)/y)*100)



    # MISC UTILITIES AND HELPERS
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
        return y_pred