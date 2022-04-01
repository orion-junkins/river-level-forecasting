import pandas as pd
import sys
from forecasting.catchment_data import CatchmentData
import pickle
from forecasting.forecaster import Forecaster
from darts.models import BlockRNNModel

model_save_dir = sys.argv[1]
model_index_to_train = sys.argv[2]

print("Training model ", model_index_to_train)

overwrite_existing_models = False 

best_params = {
        "input_chunk_length" : 120,
        "output_chunk_length" : 96
    }

rebuild_catchment = True
if rebuild_catchment:
    catchment = CatchmentData("illinois-kerby", "14377100")

    pickle_out = open("data/catchment.pickle", "wb")
    pickle.dump(catchment, pickle_out)
    pickle_out.close()
else:
    pickle_in = open("data/catchment.pickle", "rb")
    catchment = pickle.load(pickle_in)

forecaster = Forecaster(catchment, 
                                model_type=BlockRNNModel, 
                                model_params=best_params, 
                                model_save_dir=model_save_dir,
                                overwrite_existing_models=overwrite_existing_models)

forecaster.fit(epochs=20, model_indexes=[model_index_to_train])
