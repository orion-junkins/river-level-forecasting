# This script is intended as a demo of how to use the evaluator to evaluate past inference results. This is useful for evaluating both NOAA cached forecasts and forecasts generated on cached weather data.
# %% Setup
from rlf.evaluating.evaluator import build_evaluator_from_csv

# Toggle False if running from the command
running_as_notebook = True
if running_as_notebook:
    evaluator = build_evaluator_from_csv(path="../flow_summary copy 2.csv")
else:
    evaluator = build_evaluator_from_csv(path="data/inference_eval_example.csv")

# %% Calculate the mean absolute error for each window size.
mae = evaluator.df_mae()
print(mae)
mae.plot()

# %% Calculate the mean absolute percentage error for each window size.
mape = evaluator.df_mape()
print(mape)
mape.plot()

# %%
