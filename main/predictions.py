import pandas as pd
import numpy as np
from keras.models import load_model
from main.utils import *
import pickle

def predict_with_lstm(model_path, scaler, sequence_length, data):
    model = load_model(model_path)
    last_sequence = data[-sequence_length:]
    predictions = []
    for _ in range(365):  # For example, predicting for the next 365 days
        current_sequence = last_sequence.reshape((1, sequence_length, 1))
        next_prediction = model.predict(current_sequence)[0][0]
        predictions.append(next_prediction)
        last_sequence = np.roll(last_sequence, -1)
        last_sequence[-1] = next_prediction
    return scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()

def predict_with_rnn(model_path, scaler, sequence_length, data):
    model = load_model(model_path)
    last_sequence = data[-sequence_length:]
    predictions = []
    for _ in range(365):  # Predicting for the next 365 days
        current_sequence = last_sequence.reshape((1, sequence_length, 1))
        next_prediction = model.predict(current_sequence)[0][0]
        predictions.append(next_prediction)
        last_sequence = np.roll(last_sequence, -1)
        last_sequence[-1] = next_prediction
    return scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()


def predict_with_prophet(model_path, future_periods=365):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    future = model.make_future_dataframe(periods=future_periods)
    forecast = model.predict(future)
    return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

def main():
    # Load and preprocess data
    df = load_data('/path/to/your/data.csv')
    df = feature_engineering(df)

    # Predict with LSTM
    scaled_data, _, scaler = scale_data(df, df)  # Assuming entire data is used for scaling
    lstm_predictions = predict_with_lstm('/Users/ekanshtrivedi/Fetch-MLE/LSTM.h5', scaler, 60, scaled_data)

    # Predict with RNN
    scaled_data, _, scaler = scale_data(df, df)  # Assuming entire data is used for scaling
    rnn_predictions = predict_with_rnn('/Users/ekanshtrivedi/Fetch-MLE/RNN.h5', scaler, 60, scaled_data)

    # Predict with Prophet
    prophet_forecast = predict_with_prophet('/Users/ekanshtrivedi/Fetch-MLE/prophet.pkl')


if __name__ == "__main__":
    main()
