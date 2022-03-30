import pickle
import sys
import pandas as pd
from forecasting.catchment_data import CatchmentData
from forecasting.forecaster import Forecaster

horizon = int(sys.argv[1])
stride = int(sys.argv[2])

pickle_in = open("data/catchment.pickle", "rb")
catchment = pickle.load(pickle_in)


best_params = {"input_chunk_length" : 120,"output_chunk_length" : 96}


from darts.models import NBEATSModel
forecaster = Forecaster(catchment, 
                                model_type=NBEATSModel, 
                                model_params=best_params, 
                                model_save_dir="NBeats",
                                overwrite_existing_models=False)

hst_fcasts = forecaster.historical_forecasts(forecast_horizon=horizon, stride=stride, num_samples=100)

print(hst_fcasts)

pickle_out = open(f"historical_forecasts/h{horizon}_s{stride}.pickle", "wb")
pickle.dump(hst_fcasts, pickle_out)
pickle_out.close()