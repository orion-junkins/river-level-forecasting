#%%
import pickle
from forecasting.model_builders import build_conv_model
from forecasting.data_fetcher import DataFetcher
from forecasting.forecaster import Forecaster
from fsite_data import *

#%%
refetch_data = False  
if refetch_data:
    illinois_kerby = DataFetcher("14377100", illinois_kerby_weather_sources  )
    pickle_out = open("fsites/illinois_kerby.pickle", "wb")
    pickle.dump(illinois_kerby, pickle_out)
    pickle_out.close()
else:
    pickle_in = open("fsites/illinois_kerby.pickle", "rb")
    illinois_kerby = pickle.load(pickle_in)

#%%
frcstr = Forecaster(illinois_kerby, build_conv_model)
#%%
retrain = True
if retrain:
    frcstr.fit(epochs=50)
else: 
    frcstr.load_trained()
  # %%
frcstr.forecast_for("2022-02-01 16:00:00")
# %%
model = frcstr.model
X_test = frcstr.dataset.X_test_shaped
y_test = frcstr.dataset.y_test
# %%
y_pred = model.predict(X_test)
y_pred = frcstr.dataset.target_scaler.inverse_transform(y_pred)
y_true = frcstr.dataset.target_scaler.inverse_transform(y_test)
# %%
plt.plot(y_true[5000:6000], color='blue')
plt.plot(y_pred[5000:7000], color="green")
plt.show()