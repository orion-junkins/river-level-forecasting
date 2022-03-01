import streamlit as st
import pickle
import pandas as pd
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

data_load_state = st.text('Loading data...')

st.subheader('24 hour advance prediction window')
hst_fcasts_24_4 = test_historical(24, 4)
st.line_chart(hst_fcasts_24_4)

st.subheader('48 hour advance prediction window')
hst_fcasts_48_4 = test_historical(48, 4)
st.line_chart(hst_fcasts_48_4)

st.subheader('72 hour advance prediction window')
hst_fcasts_72_4 = test_historical(72, 4)
st.line_chart(hst_fcasts_72_4)