#%%
import os
import sys
import pickle

from forecasting.general_utilities.aws_utils import pickle_to_aws

# Provide name of model and catchment. 
MODEL_NAME = "GRU_RandomForest_24_0" #sys.argv[1]
CATCHMENT_NAME = "illinois-kerby"

# Specify duration of forecast desired
HOURS_TO_FORECAST = 24

# Define paths to access trained forecaster
TRAINED_MODEL_DIR = "trained_models"
ENSEMBLE_MODEL_DIR = os.path.join(TRAINED_MODEL_DIR, CATCHMENT_NAME, MODEL_NAME)
FRCSTR_FILE = os.path.join(ENSEMBLE_MODEL_DIR, MODEL_NAME + "_frcstr.pickle")

# Load trained forecaster
pickle_in = open(FRCSTR_FILE, "rb")
frcstr = pickle.load(pickle_in)

# Update the forecaster to grab up to date data
# frcstr.update_input_data()

# Generate forecast
y_forecasted = frcstr.forecast_for_hours(HOURS_TO_FORECAST)

# Grab recent data
y_recent = frcstr.dataset.y_current # abstract into Forecaster.recent_level()
target_scaler = frcstr.dataset.target_scaler
y_recent = target_scaler.inverse_transform(y_recent)
y_recent = y_recent.pd_dataframe()
#%%
# Dispatch data to AWS as .pickle files
pickle_to_aws(y_recent, river_gauge_name=CATCHMENT_NAME, file_prefix="recent")
pickle_to_aws(y_forecasted, river_gauge_name=CATCHMENT_NAME, file_prefix="forecasted")
