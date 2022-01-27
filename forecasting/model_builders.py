import tensorflow as tf
from tensorflow.keras import layers


def build_conv_model(input_shape):
    """
    Helper function for building a simple 2D convolution model

    Args:
        input_shape (tuple): input shape of data

    Returns:
        model: untrained, compiled tf.keras model
    """
    model = tf.keras.Sequential()

    model.add(layers.Conv2D(16, (2,2), input_shape=input_shape))
    model.add(layers.Dropout(0.1))
    model.add(layers.Conv2D(32, (2,2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    model.compile(loss='mse', optimizer='adam')

    return model

def build_conv_LSTM_model(input_shape):
    """
    Helper function for building a simple 2D convolution LSTM model

    Args:
        input_shape (tuple): input shape of data

    Returns:
        model: untrained, compiled tf.keras model
    """
    model = tf.keras.Sequential()

    model.add(layers.ConvLSTM2D(16, 2, input_shape=input_shape, padding='same', return_sequences=True))
    model.add(layers.Dropout(0.1))
    model.add(layers.ConvLSTM2D(32, 2, input_shape=input_shape, padding='same', return_sequences=True))
    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    model.compile(loss='mse', optimizer='adam')

    return model