import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from pandas.tseries.holiday import USFederalHolidayCalendar

def load_data(file_path):
    """Loads data from a CSV file."""
    return pd.read_csv(file_path)

def feature_engineering(df):
    """Performs feature engineering on the DataFrame."""
    cal = USFederalHolidayCalendar()
    holidays = cal.holidays(start=df['# Date'].min(), end=df['# Date'].max())
    df['# Date'] = pd.to_datetime(df['# Date'])
    df['month'] = df['# Date'].dt.month
    df['day'] = df['# Date'].dt.day
    df['day_of_week'] = df['# Date'].dt.dayofweek
    df['quarter'] = df['# Date'].dt.quarter
    df['is_weekend'] = df['# Date'].dt.dayofweek >= 5
    df['is_holiday'] = df['# Date'].isin(holidays).astype(int)
    return df

def scale_data(train, test):
    """Scales the train and test data."""
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_train = scaler.fit_transform(train[['Receipt_Count']])
    scaled_test = scaler.transform(test[['Receipt_Count']])
    return scaled_train, scaled_test, scaler

def create_sequences(data, sequence_length):
    """Creates sequences of data for LSTM or RNN models."""
    X, y = [], []
    for i in range(sequence_length, len(data)):
        X.append(data[i - sequence_length:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

def split_data(df, split_date):
    """Splits the data into training and testing sets."""
    train = df.loc[df['# Date'] < split_date]
    test = df.loc[df['# Date'] >= split_date]
    return train, test

def prepare_features(train, test):
    """Prepares training and testing features."""
    X_train = train[['month', 'day', 'day_of_week', 'quarter', 'is_weekend', 'is_holiday']]
    y_train = train['Receipt_Count']
    X_test = test[['month', 'day', 'day_of_week', 'quarter', 'is_weekend', 'is_holiday']]
    y_test = test['Receipt_Count']
    return X_train, y_train, X_test, y_test
