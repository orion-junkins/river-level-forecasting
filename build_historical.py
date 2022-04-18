#%%
import os
import pickle
import boto3

IN_PATH = os.path.join("data", "forecast_ensemble.pickle")
OUT_PATH = os.path.join("data", "historical_forecast_df.pickle")

pickle_in = open(IN_PATH, "rb")
frcstr = pickle.load(pickle_in)

hst_fcasts = frcstr.historical_forecasts()

pickle_out = open(OUT_PATH, "wb")
pickle.dump(hst_fcasts, pickle_out)
pickle_out.close()
#%%
def pickle_to_aws(y, river_gauge_name, file_prefix):
    pickle_out = open(OUT_PATH, "wb")
    pickle.dump(y, pickle_out)
    pickle_out.close()
    s3 = boto3.client('s3')
    s3.upload_file(OUT_PATH, f'generated-forecasts', f'{river_gauge_name}/{file_prefix}_dataframe.pickle')

