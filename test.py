#%%
from forecasting.data_fetching_utilities.weather import *

test = {
(41.980609,-123.613583): "historical_weather_data/illinois-kerby/41.980609,-123.613583.csv"
}

df = fetch_one_call(41.980609,-123.613583)
df
type(df)
df.columns
# %%
