from keras.models import Model
from keras.layers import Input, Conv2D, Flatten, MaxPooling2D
from keras.layers.merge import concatenate
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM

def structure(timestep, dimension, row, col):
    model_lstm_in = Input(shape=(timestep, dimension))
    model_lstm = LSTM(units=64, return_sequences=True)(model_lstm_in)
    model_lstm_out = LSTM(units=64)(model_lstm)

    model_cnn_in = Input(shape=(row, col, 1))
    model_cnn = Conv2D(filters=10, kernel_size=3, activation="relu")(model_cnn_in)
    model_cnn = MaxPooling2D(pool_size=2)(model_cnn)
    model_cnn = Conv2D(filters=10, kernel_size=3, activation="relu")(model_cnn)
    model_cnn = MaxPooling2D(pool_size=2)(model_cnn)
    model_cnn = Conv2D(filters=10, kernel_size=3, activation="relu")(model_cnn)
    model_cnn_out = Flatten()(model_cnn)

    concatenate_model = concatenate([model_lstm_out, model_cnn_out])

    concatenate_model = Dense(32, activation="relu")(concatenate_model)
    concatenate_model = Dense(16, activation="relu")(concatenate_model)
    out = Dense(units=6, activation="softmax")(concatenate_model)

    merged_model = Model([model_lstm_in, model_cnn_in], out)

    print(merged_model.summary())

    return merged_model