import os
import pickle

from forecasting.catchment_data import CatchmentData

CATCHMENT_NAME = "illinois-kerby"
OUT_DIR_PATH = os.path.join("data", "catchments", CATCHMENT_NAME)
os.makedirs(OUT_DIR_PATH, exist_ok=True)
OUT_FILE_PATH = os.path.join(OUT_DIR_PATH, "catchment.pickle")

gauge_ids = {
    "illinois-kerby": "14377100"
}

catchment = CatchmentData(CATCHMENT_NAME, gauge_ids[CATCHMENT_NAME])


pickle_out = open(OUT_FILE_PATH, "wb")
pickle.dump(catchment, pickle_out)
pickle_out.close()
