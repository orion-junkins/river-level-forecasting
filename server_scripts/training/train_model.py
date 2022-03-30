import pandas as pd
import sys
from forecasting.catchment_data import CatchmentData
import pickle
from forecasting.forecaster import Forecaster
from darts.models import NBEATSModel

model_save_dir = sys.argv[1]
model_index_to_train = sys.argv[2]

print("Training BlockRNN model ", model_index_to_train)

overwrite_existing_models = False 

best_params = {
        "input_chunk_length" : 120,
        "output_chunk_length" : 96
    }

pickle_in = open("data/catchment.pickle", "rb")
catchment = pickle.load(pickle_in)

forecaster = Forecaster(catchment, 
                                model_type=NBEATSModel, 
                                model_params=best_params, 
                                model_save_dir=model_save_dir,
                                overwrite_existing_models=overwrite_existing_models)

forecaster.fit(epochs=20, model_indexes=[model_index_to_train])
