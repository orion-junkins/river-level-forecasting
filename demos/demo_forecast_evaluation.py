# This script is intended as a demo of how to use the evaluator to evaluate past inference results. This is useful for evaluating both NOAA cached forecasts and forecasts generated on cached weather data.
# %% Setup
from rlf.evaluating.evaluator import build_evaluator_from_csv

# Toggle False if running from the command
running_as_notebook = True
if running_as_notebook:
    custom_evaluator = build_evaluator_from_csv(path="../inference_simulations/500_12143400_geo_2.csv")
    nws_evaluator = build_evaluator_from_csv(path="C:/Users/orion/Documents/GitHub/noaa-hydro-forecast-evaluation/data/processed/nwrfc-0/GARW1/flow_summary.csv")
else:
    evaluator = build_evaluator_from_csv(path="data/inference_eval_example.csv")

#%%
custom_evaluator.filter_timestamps(nws_evaluator)
nws_evaluator.filter_timestamps(custom_evaluator)
#%%
df_custom = custom_evaluator.df_mape().rename(columns={0: "Custom Model"})
df_nws = nws_evaluator.df_mape().rename(columns={0: "NWS Model"})

# Concat the two dataframes
df = df_custom.join(df_nws).dropna()
df.plot()



# %% Calculate the mean absolute error for each window size.
mae = evaluator.df_mae()
print(mae)
mae.plot()

# %% Calculate the mean absolute percentage error for each window size.
mape = evaluator.df_mape()
print(mape)
mape.plot()

# %%

# %%
