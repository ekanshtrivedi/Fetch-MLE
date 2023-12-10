import pandas as pd
import numpy as np
from keras.models import load_model
from utils import *
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
    start_date = '2022-01-01'
    end_date = '2022-12-31'
    # Load and preprocess data
    df = load_data('/Users/ekanshtrivedi/Fetch-MLE/data_daily.csv')
    df = feature_engineering(df)

    # Predict with LSTM
    scaled_data, _, scaler = scale_data(df, df)  # Assuming entire data is used for scaling
    lstm_predictions = predict_with_lstm('/Users/ekanshtrivedi/Fetch-MLE/trained_models/LSTM.h5', scaler, 60, scaled_data)
    lstm_df = resample_predictions(lstm_predictions, start_date, end_date, freq='M')
    lstm_df.to_csv('/Users/ekanshtrivedi/Fetch-MLE/results/lstm_predictions1.csv')

    # Predict with RNN
    scaled_data, _, scaler = scale_data(df, df)  # Assuming entire data is used for scaling
    rnn_predictions = predict_with_rnn('/Users/ekanshtrivedi/Fetch-MLE/trained_models/RNN.h5', scaler, 60, scaled_data)
    rnn_df = resample_predictions(rnn_predictions, start_date, end_date, freq='M')
    rnn_df.to_csv('/Users/ekanshtrivedi/Fetch-MLE/results/rnn_predictions1.csv')

    
    # Predict with Prophet
    prophet_forecast = predict_with_prophet('/Users/ekanshtrivedi/Fetch-MLE/trained_models/prophet.pkl')
    prophet_forecast['ds'] = pd.to_datetime(prophet_forecast['ds'])
    prophet_forecast.set_index('ds', inplace=True)
    monthly_forecast = prophet_forecast.resample('M').sum()
    forecast_2022 = monthly_forecast.loc['2022']
    monthly_predictions_prophet = forecast_2022[['yhat']]
    monthly_predictions_prophet = monthly_predictions_prophet.reset_index()
    monthly_predictions_prophet = monthly_predictions_prophet.rename(columns={'ds': 'Date', 'yhat': 'Prediction'})
    monthly_predictions_prophet.to_csv('/Users/ekanshtrivedi/Fetch-MLE/results/prophet_predictions1.csv', index=False)



if __name__ == "__main__":
    main()
