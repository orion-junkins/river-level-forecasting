#%%
import pickle
from forecasting.model_builders import build_conv_model
from forecasting.forecast_site import ForecastSite
from forecasting.forecaster import Forecaster

rebuild_fsites = False  
if rebuild_fsites:
    illinois_kerby = ForecastSite("14377100", ["historical_weather_data/illinois-kerby/41.980609,-123.613583.csv"])
    pickle_out = open("fsites/illinois_kerby.pickle", "wb")
    pickle.dump(illinois_kerby, pickle_out)
    pickle_out.close()
else:
    pickle_in = open("fsites/illinois_kerby.pickle", "rb")
    illinois_kerby = pickle.load(pickle_in)

# %%

# %%
frcstr = Forecaster([mck1_weather_url, mck2_weather_url], mck_vida_url, build_conv_model)

frcstr_backup = frcstr
#%%
frcstr.fit()

#%%
frcstr.forecast_all()
print(frcstr.forecasted_levels)