import sys
import pickle 
from forecasting.catchment_data import CatchmentData
from forecasting.dataset import Dataset
from darts.models import NBEATSModel

outfile_name = sys.argv[1]
pickle_in = open("data/catchment.pickle", "rb")
catchment = pickle.load(pickle_in)

dataset = Dataset(catchment)
pl_trainer_kwargs={
        "accelerator": "gpu",
        "gpus": [0]
    }

gridsearch_params = {
    "pl_trainer_kwargs" : [pl_trainer_kwargs],
    "input_chunk_length" : [120, 144],
    "output_chunk_length" : [96],
    "num_stacks" : [25, 50],
    "num_blocks" : [1, 2, 3, 5, 10],
    "num_layers" : [2, 3, 4],
    "layer_widths" : [256, 512, 1024]
}

X_train = dataset.X_trains[0]
y_train = dataset.y_train

y_train, y_val = y_train.split_before(0.7)

model, best_params = NBEATSModel.gridsearch(gridsearch_params, verbose=True, series=y_train, past_covariates=X_train, val_series=y_val)

print(best_params)

pickle_out = open(f"trained_models/grid_search_output/{outfile_name}_params.pickle", "wb")
pickle.dump(best_params, pickle_out)
pickle_out.close()

pickle_out = open(f"trained_models/grid_search_output/{outfile_name}_model.pickle", "wb")
pickle.dump(model, pickle_out)
pickle_out.close()