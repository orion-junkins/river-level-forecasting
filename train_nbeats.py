import pandas as pd
import sys
from forecasting.catchment_data import CatchmentData
import pickle
from forecasting.forecaster import Forecaster
from darts.models import NBEATSModel

model_index_to_train = int(sys.argv[1])
print("Training NBEATS model ", model_index_to_train)

rebuild_catchment = False
overwrite_existing_models = True
best_params = {
        "input_chunk_length" : 120,
        "output_chunk_length" : 96
    }

if rebuild_catchment:
    catchment = CatchmentData("illinois-kerby", "14377100")

    pickle_out = open("temp_storage/catchment.pickle", "wb")
    pickle.dump(catchment, pickle_out)
    pickle_out.close()
else:
    pickle_in = open("temp_storage/catchment.pickle", "rb")
    catchment = pickle.load(pickle_in)


forecaster = Forecaster(catchment, 
                                model_type=NBEATSModel, 
                                model_params=best_params, 
                                model_save_dir="NBeats2",
                                overwrite_existing_models=overwrite_existing_models)

forecaster.fit(epochs=10, model_indexes=[model_index_to_train])


def build_historical(forecaster, horizon, stride, name="test"):
    y_pred = forecaster.historical_forecasts(forecast_horizon=horizon, stride=stride, num_samples=100)
    y_true = forecaster.dataset.y_test
    target_scaler = forecaster.dataset.target_scaler
    y_true = target_scaler.inverse_transform(y_true)
    y_true = y_true.pd_dataframe()
    all_historical = pd.concat([y_true, y_pred], join='inner', axis=1)
    pickle_out = open(f"temp_storage/historical/NBeats/illinois-kerby/{name}/w{horizon}_s{stride}.pickle", "wb")
    pickle.dump(all_historical, pickle_out)
    pickle_out.close()

build_historical(forecaster, 24, 4)
build_historical(forecaster, 48, 4)
build_historical(forecaster, 72, 4)

forecaster.fit(epochs=10)

build_historical(forecaster, 24, 4, name='test2')
build_historical(forecaster, 48, 4, name='test2')
build_historical(forecaster, 72, 4, name='test2')