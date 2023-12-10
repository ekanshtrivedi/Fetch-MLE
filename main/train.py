import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import LSTM, Dense, SimpleRNN, Dropout
from keras.optimizers import Adam
from prophet import Prophet
from utils import *


def train_linear_regression(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def train_lstm_model(X_train, y_train, n_layers, units, dropout_rate, learning_rate):
    model = Sequential()
    model.add(LSTM(units=units, return_sequences=True if n_layers > 1 else False, input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(dropout_rate))
    for i in range(1, n_layers):
        model.add(LSTM(units=units, return_sequences=(i < n_layers - 1)))
        model.add(Dropout(dropout_rate))
    model.add(Dense(1))
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=100, batch_size=64)
    return model

def train_rnn_model(X_train, y_train, n_units, learning_rate):
    model = Sequential()
    model.add(SimpleRNN(units=n_units, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(SimpleRNN(units=n_units))
    model.add(Dense(1))
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=100, batch_size=32)
    return model

def train_prophet_model(df_prophet):
    model = Prophet()
    model.fit(df_prophet)
    return model

def main():
    # Load and preprocess data
    df2 = load_data('/Users/ekanshtrivedi/Fetch-MLE/data_daily.csv')
    df2 = feature_engineering(df2)
    train, test = split_data(df2, split_date='2021-09-01')
    X_train, y_train, X_test, y_test = prepare_features(train, test)
    scaled_train, scaled_test, scaler = scale_data(train, test)
    X_lstm, y_lstm = create_sequences(scaled_train, sequence_length=60)
    X_lstm = np.reshape(X_lstm, (X_lstm.shape[0], X_lstm.shape[1], 1))


    # Train models
    lr_model = train_linear_regression(X_train, y_train)
    lstm_model = train_lstm_model(X_lstm, y_lstm, 2, 50, 0.2, 0.001)
    rnn_model = train_rnn_model(X_lstm, y_lstm, 50, 0.001)
    prophet_model = train_prophet_model(df2.rename(columns={'# Date': 'ds', 'Receipt_Count': 'y'}))


if __name__ == "__main__":
    main()
