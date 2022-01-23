import tensorflow as tf
from tensorflow.keras import layers
def build_conv_model(input_shape):
    # Helper function for building a basic convolutional NN model
    # Create the model
    model = tf.keras.Sequential()

    # Add desired layers
    model.add(layers.Conv2D(16, (2,2), input_shape=input_shape))
    model.add(layers.Dropout(0.1))
    model.add(layers.Conv2D(32, (2,2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    # Compile the model
    model.compile(loss='mse', optimizer='adam')

    # Return the model
    return model