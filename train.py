#%%
import os
import sys
import pickle
from darts.models import BlockRNNModel
from forecasting.forecaster import Forecaster
from sklearn.svm import SVR

reg_models = [SVR()]
# Provide name of model and catchment. These will determine model save directory names.
MODEL_NAME = "TEST5-7"
CATCHMENT_NAME = "illinois-kerby"

# Define root directory for trained models
TRAINED_MODEL_DIR = "trained_models"

# Define specific paths for catchment and ensemble models
CATCHMENT_MODEL_DIR = os.path.join(TRAINED_MODEL_DIR, CATCHMENT_NAME, MODEL_NAME, "catchment_models")
ENSEMBLE_MODEL_DIR = os.path.join(TRAINED_MODEL_DIR, CATCHMENT_NAME, MODEL_NAME)
FRCSTR_OUTFILE = os.path.join(ENSEMBLE_MODEL_DIR, MODEL_NAME + "_frcstr.pickle")

# Ensure directories exist
os.makedirs(CATCHMENT_MODEL_DIR, exist_ok=True)
os.makedirs(ENSEMBLE_MODEL_DIR, exist_ok=True)

# Specify where to fetch catchment data
CATCHMENT_DATA_FILEPATH = os.path.join("data", "catchments", CATCHMENT_NAME, "catchment.pickle")
pickle_in = open(CATCHMENT_DATA_FILEPATH, "rb")
catchment = pickle.load(pickle_in)

# Define model type for catchment models
model_builder = BlockRNNModel

# Define parameters for model
best_params = {'pl_trainer_kwargs': {'accelerator': 'gpu', 'gpus': [0]}, 
    'n_epochs': 20, 
    'input_chunk_length': 120, 
    'output_chunk_length': 6, 
    'model': 'GRU', 
    'hidden_size': 50, 
    'n_rnn_layers': 5, 
    'dropout': 0.01}


test_params = {'pl_trainer_kwargs': {'accelerator': 'gpu', 'gpus': [0]}, 
    'n_epochs': 0, 
    'input_chunk_length': 6, 
    'output_chunk_length': 6, 
    'model': 'GRU', 
    'hidden_size': 5, 
    'n_rnn_layers': 1, 
    'dropout': 0.01}


# Create desired number of catchment models (must be the same as the number of catchments)
NUM_MODELS = 12
catchment_models = []
for i in range(NUM_MODELS):
    model = model_builder(work_dir=CATCHMENT_MODEL_DIR, 
                            model_name=str(i), 
                            force_reset=False, 
                            save_checkpoints=True, 
                            **test_params)
    catchment_models.append(model)

# Create forecaster
frcstr = Forecaster(catchment, catchment_models)
#%%
# Fit the forecaster
frcstr.fit(epochs=0)

# Fit additional models
for reg_model in reg_models:
    if reg_model not in frcstr.regression_models:
        frcstr.fit(reg_model=reg_model)

#%%
# Save the trained forecaster 
pickle_out = open(FRCSTR_OUTFILE, "wb")
pickle.dump(frcstr, pickle_out)
pickle_out.close()

preds = frcstr.predict()

# # Print out a sample forecast
print(preds)


# %%
