import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from pandas.tseries.holiday import USFederalHolidayCalendar

def load_data(file_path):
    """
    Loads data from a specified CSV file.

    Args:
    file_path (str): Path to the CSV file.

    Returns:
    DataFrame: Loaded data as a pandas DataFrame.
    """
    return pd.read_csv(file_path)

def feature_engineering(df):
    """
    Performs feature engineering on the DataFrame.

    Args:
    df (DataFrame): The original DataFrame.

    Returns:
    DataFrame: The DataFrame with additional engineered features.
    """
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
    """
    Scales the train and test data using MinMaxScaler.

    Args:
    train (DataFrame): Training data.
    test (DataFrame): Testing data.

    Returns:
    tuple: Scaled training data, scaled testing data, and the scaler instance.
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_train = scaler.fit_transform(train[['Receipt_Count']])
    scaled_test = scaler.transform(test[['Receipt_Count']])
    return scaled_train, scaled_test, scaler

def create_sequences(data, sequence_length):
    """
    Creates sequences of data for LSTM or RNN models.

    Args:
    data (array-like): Time series data.
    sequence_length (int): Length of the sequence.

    Returns:
    tuple: Sequences of features (X) and target (y).
    """
    X, y = [], []
    for i in range(sequence_length, len(data)):
        X.append(data[i - sequence_length:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

def resample_predictions(daily_predictions, start_date, end_date, freq='M'):
    """
    Resamples daily predictions to a specified frequency.

    Args:
    daily_predictions (array-like): Array of daily predictions.
    start_date (str): Start date of the prediction range.
    end_date (str): End date of the prediction range.
    freq (str): Frequency for resampling. Default is 'M' for monthly.

    Returns:
    DataFrame: Resampled DataFrame.
    """
    dates = pd.date_range(start=start_date, end=end_date)
    predictions_df = pd.DataFrame({'Date': dates, 'Prediction': daily_predictions})
    predictions_df.set_index('Date', inplace=True)
    return predictions_df.resample(freq).sum()

def split_data(df, split_date):
    """
    Splits the data into training and testing sets based on a split date.

    Args:
    df (DataFrame): The original DataFrame.
    split_date (str): The date to split the data on.

    Returns:
    tuple: Training and testing DataFrames.
    """
    train = df.loc[df['# Date'] < split_date]
    test = df.loc[df['# Date'] >= split_date]
    return train, test

def prepare_features(train, test):
    """
    Prepares training and testing features from the DataFrame.

    Args:
    train (DataFrame): Training data.
    test (DataFrame): Testing data.

    Returns:
    tuple: Features and target for training and testing sets.
    """
    X_train = train[['month', 'day', 'day_of_week', 'quarter', 'is_weekend', 'is_holiday']]
    y_train = train['Receipt_Count']
    X_test = test[['month', 'day', 'day_of_week', 'quarter', 'is_weekend', 'is_holiday']]
    y_test = test['Receipt_Count']
    return X_train, y_train, X_test, y_test
