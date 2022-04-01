#%%
import pickle
from forecasting.catchment_data import CatchmentData
from forecasting.forecaster import Forecaster

rebuild_catchment = False
if rebuild_catchment:
    catchment = CatchmentData("illinois-kerby", "14377100")

    pickle_out = open("data/catchment.pickle", "wb")
    pickle.dump(catchment, pickle_out)
    pickle_out.close()
else:
    pickle_in = open("data/catchment.pickle", "rb")
    catchment = pickle.load(pickle_in)

best_params = {"input_chunk_length" : 120,"output_chunk_length" : 96}


# catchment.update_for_inference()
from darts.models import BlockRNNModel
forecaster = Forecaster(catchment, 
                                model_type=BlockRNNModel, 
                                model_params=best_params, 
                                model_save_dir="BlockckRNN2",
                                overwrite_existing_models=False)


fcast = forecaster.forecast_for_hours(n=72)

print(fcast)
