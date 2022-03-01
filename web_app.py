from distutils.command.build import build
import streamlit as st
import pickle
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

from darts.models import BlockRNNModel
from forecasting.catchment_data import CatchmentData
from forecasting.forecaster import Forecaster



@st.cache
def test_historical(horizon, stride):
    pickle_in = open("temp_storage/catchment.pickle", "rb")
    catchment = pickle.load(pickle_in)
    block_rnn_forecaster = Forecaster(catchment, model_type=BlockRNNModel, model_save_dir="BlockRNN")
    y_pred = block_rnn_forecaster.historical_forecasts(forecast_horizon=horizon, stride=stride, num_samples=100)
    y_true = block_rnn_forecaster.dataset.y_validation
    target_scaler = block_rnn_forecaster.dataset.target_scaler
    y_true = target_scaler.inverse_transform(y_true)
    y_true = y_true.pd_dataframe()
    all_historical = pd.concat([y_true, y_pred], join='inner', axis=1)
    return all_historical

st.title('River Level Forecasting with ML') 


start_time, end_time = st.slider(
     "Select start time",
     min_value=datetime(2013, 1, 1, 9, 30),
     value=(datetime(2014, 1, 1, 9, 30), datetime(2015, 1, 1, 9, 30)),
     max_value=datetime(2015, 1, 1, 9, 30),
     format="MM/DD/YY - hh:mm")
st.write("Start time:", start_time)
st.write("End time:", end_time)
start = pd.to_datetime(start_time)
end = pd.to_datetime(end_time)

def build_figure(hst_fcasts, start, end):
    fig = plt.figure()
    plt.plot(hst_fcasts['max'], color='red', linewidth=0.5, label='max 0.05 confidence quantile')
    plt.plot(hst_fcasts['mean'], color='red', linewidth=2, label='Predicted value')
    plt.plot(hst_fcasts['min'], color='red', linewidth=0.5, label='min 0.05 confidence quantile')

    plt.fill_between(hst_fcasts.index, y1=hst_fcasts['min'], y2=hst_fcasts['max'], color='red', alpha=0.3)

    plt.plot(hst_fcasts.index, hst_fcasts['level'],  color='blue', linewidth=2, label='Actual value')
    plt.legend()

    y_lim = hst_fcasts.loc[start:end, :]['max'].max() * 1.2
    plt.xlim(start,end)
    plt.ylim(-100, y_lim)
    return fig

st.subheader('24 hour advance prediction window')
hst_fcasts_24_4 = test_historical(24, 4)
st.pyplot(build_figure(hst_fcasts_24_4, start, end))

st.subheader('48 hour advance prediction window')
hst_fcasts_48_4 = test_historical(48, 4)
st.pyplot(build_figure(hst_fcasts_48_4, start, end))

st.subheader('72 hour advance prediction window')
hst_fcasts_72_4 = test_historical(72, 4)
st.pyplot(build_figure(hst_fcasts_72_4))