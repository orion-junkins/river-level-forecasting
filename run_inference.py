import os
import pickle
from forecasting.general_utilities.aws_utils import pickle_to_aws

# Provide name of model
MODEL_NAME = "RNN_GRU_12_6_MLP"

# Provide name of catchment
CATCHMENT_NAME = "illinois-kerby"

# Specify duration of forecast desired
HOURS_TO_FORECAST = 96

# Define paths to access trained forecaster
TRAINED_MODEL_DIR = "trained_models"
ENSEMBLE_MODEL_DIR = os.path.join(TRAINED_MODEL_DIR, CATCHMENT_NAME, MODEL_NAME)
FRCSTR_FILE = os.path.join(ENSEMBLE_MODEL_DIR, MODEL_NAME + "_frcstr.pickle")

# Load trained forecaster
pickle_in = open(FRCSTR_FILE, "rb")
frcstr = pickle.load(pickle_in)

# Get forecast
cur_fcast = frcstr.get_forecast(hours_to_forecast=HOURS_TO_FORECAST, update_dataset=True)

# Store the produced forecast locally and dispatch to AWS
pickle_to_aws(cur_fcast, river_gauge_name=CATCHMENT_NAME, model_name=MODEL_NAME, filename="current_forecast")