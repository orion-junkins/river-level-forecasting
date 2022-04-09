#%%
import os
import pickle
import boto3

IN_PATH = os.path.join("data", "forecast_ensemble.pickle")
OUT_PATH = os.path.join("data", "forecast_df.pickle")

pickle_in = open(IN_PATH, "rb")
frcstr = pickle.load(pickle_in)

y_pred = frcstr.forecast_for_hours(96)
print(y_pred)

# %%
pickle_out = open(OUT_PATH, "wb")
pickle.dump(y_pred, pickle_out)
pickle_out.close()
# %%

s3 = boto3.client('s3')
s3.upload_file(OUT_PATH, 'generated-forecasts', 'forecast_dataframe.pickle')

# %%
