#%%
import os
import sys
import pickle

from darts.models import BlockRNNModel
from forecasting.catchment_data import CatchmentData
from forecasting.forecaster import Forecaster


rebuild_catchment=False # 'True' causes refetching of all data, 'False' loads cached version from .pickle file

if rebuild_catchment:
    catchment = CatchmentData("illinois-kerby", "14377100") # Specify target USGS gauge name and ID

    pickle_out = open("data/catchment.pickle", "wb")
    pickle.dump(catchment, pickle_out)
    pickle_out.close()
else:
    pickle_in = open("data/catchment.pickle", "rb")
    catchment = pickle.load(pickle_in)



model_builder = BlockRNNModel

# gridsearch optimized params for a Darts BlockRNNModel
params = {'pl_trainer_kwargs': {'accelerator': 'gpu', 'gpus': [0]}, 
    'n_epochs': 1, 
    'input_chunk_length': 120, 
    'output_chunk_length': 96, 
    'model': 'GRU', 
    'hidden_size': 50, 
    'n_rnn_layers': 5, 
    'dropout': 0.01}


# Define desired number of models (must be the same as the number of catchments)
NUM_MODELS = 12

WORK_DIR = os.path.join("trained_catchment_models", "gridsearched_gru")#sys.argv[1])

catchment_models = []

for i in range(NUM_MODELS):
    model = model_builder(work_dir=WORK_DIR, model_name=str(i), save_checkpoints=True, **params)
    catchment_models.append(model)
#%%
frcstr = Forecaster(catchment, catchment_models)

frcstr.fit()

print(frcstr.forecast_for_hours())

# %%
