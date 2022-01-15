#%%
from datetime import datetime, timedelta
from dataset import *
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
#%%
def yesterday() -> str:
    """
    Helper function for data retrieval
    Returns:
        yesterday (string): yesterdays date in the format "%Y-%m-%d"
    """
    yesterday = datetime.today()  - timedelta(days=1)
    yesterday = yesterday.strftime("%Y-%m-%d")
    return yesterday

#%% 
# Example usage: 
# Based on today's date, fetch all relevant water data
mck_vida_url = 'https://waterdata.usgs.gov/nwis/dv?cb_00060=on&format=rdb&site_no=14162500&referred_module=sw&period=&begin_date=1988-10-12&end_date=' + yesterday()

# Fetch weather & SWE data for Mckenzie basin. Atuomatically retrieves all data up to present.
mck1_weather_url ='https://wcc.sc.egov.usda.gov/reportGenerator/view_csv/customSingleStationReport/daily/619:OR:SNTL%7Cid=%22%22%7Cname/POR_BEGIN,POR_END/WTEQ::value,PREC::value,TMAX::value,TMIN::value,TAVG::value,PRCP::value'

mck2_weather_url = 'https://wcc.sc.egov.usda.gov/reportGenerator/view_csv/customMultiTimeSeriesGroupByStationReport/daily/start_of_period/719:OR:SNTL%7Cid=%22%22%7Cname/1980-11-03,2022-01-08/WTEQ::value,TMAX::value,TMIN::value,TAVG::value,PRCP::value?fitToScreen=false'

ds = Dataset([mck1_weather_url, mck2_weather_url], mck_vida_url)
# %%
def build_conv_model():
    # Helper function for building a basic convolutional NN model
    # Create the model
    model = tf.keras.Sequential()

    # Add desired layers
    model.add(layers.Conv2D(16, (2,2), input_shape=X_train[0].shape))
    model.add(layers.Dropout(0.1))
    model.add(layers.Conv2D(32, (2,2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    # Compile the model
    model.compile(loss='mse', optimizer='adam')

    # Return the model
    return model


# %%
# Add an extra dimension to match input_shape of Conv2D model
X_train = ds.X_train.reshape(ds.X_train.shape[0], ds.X_train.shape[1], ds.X_train.shape[2], 1)
X_test = ds.X_test.reshape(ds.X_test.shape[0], ds.X_test.shape[1], ds.X_test.shape[2], 1)

# Build a model
basic_conv_model = build_conv_model()

# Fit the model
basic_conv_model.fit(X_train, ds.y_train, epochs = 20, batch_size = 10, 
                        shuffle = True)
# %%
def plot_predictions(y_pred, y_true, start_index=0, end_index=-1):
    plt.plot(y_pred[start_index:end_index], label='y_pred')
    plt.plot(y_true[start_index:end_index], label='y_true')
    plt.legend()
    plt.title("Predicted Level and True Level over time")
    plt.xlabel("Day in Range")
    plt.ylabel("Level (cubic feet per second)")
    plt.show()

def evaluate_model(model, target_scaler, X_test, y_test):
    # Print a general model evaluation in the given X, y sets
    print("Test set loss:")
    print(model.evaluate(x=X_test, y=y_test))
    print()
    print()

    # Generate raw predictions for the given X test data
    y_pred = np.array(model(X_test))

    # Inverse transfrom y_pred and y_test values back into cfs (undo MinMax scaling)
    y_pred = target_scaler.inverse_transform(y_pred)
    y_true = target_scaler.inverse_transform(y_test.reshape(-1, 1))

    # Plot the predictions for the past 40 days
    print("Predictions and true values for past 40 days:")
    plot_predictions(y_pred, y_true, start_index=-40)
    print()

    # Plot the predictions for the past 100 days
    print("Predictions and true values for past 100 days:")
    plot_predictions(y_pred, y_true, start_index=-100)
#%%

drid# %%
