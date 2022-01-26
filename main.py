#%%
import pickle
from forecasting.model_builders import build_conv_model
from forecasting.forecast_site import ForecastSite
from forecasting.forecaster import Forecaster
from fsite_data import *

#%%
rebuild_fsites = False  
if rebuild_fsites:
    illinois_kerby = ForecastSite("14377100", illinois_kerby_weather_sources)
    pickle_out = open("fsites/illinois_kerby.pickle", "wb")
    pickle.dump(illinois_kerby, pickle_out)
    pickle_out.close()
else:
    pickle_in = open("fsites/illinois_kerby.pickle", "rb")
    illinois_kerby = pickle.load(pickle_in)

#%%
frcstr = Forecaster(illinois_kerby, build_conv_model)
frcstr.fit(epochs=20)
# %%

frcstr.forecast_for("2022-01-25 16:00:00")
# %%
