import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, GRU, Dropout, BatchNormalization


def lstm_sigmoid(train_x):
    model = Sequential()
    model.add(LSTM(128, input_shape=(train_x.shape[1:]), return_sequences=True))
    model.add(Dropout(0.25))
    model.add(BatchNormalization())  #normalizes activation outputs, same reason you want to normalize your input data.

    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(0.25))
    model.add(BatchNormalization())

    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(0.25))
    model.add(BatchNormalization())

    model.add(LSTM(128))
    model.add(Dropout(0.25))
    model.add(BatchNormalization())

    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.25))

    model.add(Dense(1, activation='sigmoid'))

    opt = tf.keras.optimizers.Adam(learning_rate=1e-4, decay=1e-6)
    # Compile model
    model.compile(optimizer=opt,
        loss='binary_crossentropy',
        metrics=['accuracy'])
    return model


def lstm_softmax(train_x):
    model = Sequential()
    model.add(LSTM(128, input_shape=(train_x.shape[1:]), return_sequences=True))
    model.add(Dropout(0.25))
    model.add(BatchNormalization())  #normalizes activation outputs, same reason you want to normalize your input data.

    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(0.25))
    model.add(BatchNormalization())

    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(0.25))
    model.add(BatchNormalization())

    model.add(LSTM(128))
    model.add(Dropout(0.25))
    model.add(BatchNormalization())

    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.25))

    model.add(Dense(2, activation='softmax'))

    opt = tf.keras.optimizers.Adam(learning_rate=1e-4, decay=1e-6)
    # Compile model
    model.compile(optimizer=opt,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])
    return model