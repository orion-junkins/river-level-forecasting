import subprocess
import sys
import pickle
from darts.models import BlockRNNModel
from forecasting.forecaster import Forecaster

pickle_in = open("data/catchment.pickle", "rb")
catchment = pickle.load(pickle_in)

model_save_dir = sys.argv[1]

best_params = {
        "input_chunk_length" : 120,
        "output_chunk_length" : 96
    }

forecaster = Forecaster(catchment, 
                                model_type=BlockRNNModel, 
                                model_params=best_params, 
                                model_save_dir=model_save_dir,
                                overwrite_existing_models=True)

processes = []

for i in range(12):
    process = subprocess.Popen(["sbatch", "/nfs/guille/eecs_research/soundbendor/junkinso/river-level-forecasting/train_model.sh", model_save_dir, str(i)])
    processes.append(process)

for process in processes:
    process.wait()