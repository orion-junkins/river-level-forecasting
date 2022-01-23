#%%
import pickle
from forecasting.model_builders import build_conv_model
from forecasting.forecast_site import ForecastSite
from forecasting.forecaster import Forecaster
from fsite_data import *

rebuild_fsites = False  
if rebuild_fsites:
    illinois_kerby = ForecastSite("14377100", illinois_kerby_weather_sources)
    pickle_out = open("fsites/illinois_kerby.pickle", "wb")
    pickle.dump(illinois_kerby, pickle_out)
    pickle_out.close()
else:
    pickle_in = open("fsites/illinois_kerby.pickle", "rb")
    illinois_kerby = pickle.load(pickle_in)

# %%
#%%
lat = 41.980609 
lon = -123.613583
from forecasting.data_fetching_utilities.weather import *
from forecasting.time_utils import *
start = unix_timestamp_days_ago(50)
end = unix_timestamp_now()
print(start)
print(end)
df = get_single_loc_recent(lat, lon, start, end)
# %%
frcstr = Forecaster([mck1_weather_url, mck2_weather_url], mck_vida_url, build_conv_model)

frcstr_backup = frcstr
#%%
frcstr.fit()

#%%
frcstr.forecast_all()
print(frcstr.forecasted_levels)