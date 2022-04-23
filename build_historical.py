import os
import pickle
from forecasting.general_utilities.aws_utils import pickle_to_aws

# Provide name of model and catchment. 
MODEL_NAME = "RNN_GRU_12_6_MLP"

# Provide name of catchment
CATCHMENT_NAME = "illinois-kerby"

# Specify desired horizon and stride
HORIZON = 24
STRIDE = 1

# Define paths to access trained forecaster
TRAINED_MODEL_DIR = "trained_models"
ENSEMBLE_MODEL_DIR = os.path.join(TRAINED_MODEL_DIR, CATCHMENT_NAME, MODEL_NAME)
FRCSTR_FILE = os.path.join(TRAINED_MODEL_DIR, CATCHMENT_NAME, MODEL_NAME, MODEL_NAME + "_frcstr.pickle")

# Load trained forecaster
pickle_in = open(FRCSTR_FILE, "rb")
frcstr = pickle.load(pickle_in)

# Produce historical forecast
hst_fcast = frcstr.get_historical(forecast_horizon=24, stride=1)

# Store the produced forecast locally and dispatch to AWS
pickle_to_aws(hst_fcast, river_gauge_name=CATCHMENT_NAME, model_name=MODEL_NAME, filename=f"historical_forecast_h{HORIZON}_s{STRIDE}")