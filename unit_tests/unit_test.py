import unittest
from main.utils import load_data, feature_engineering, scale_data, create_sequences, split_data, prepare_features
import pandas as pd

class TestUtils(unittest.TestCase):
    """
    Test suite for validating utility functions in the project.
    """

    def test_load_data(self):
        """
        Test whether the load_data function correctly loads data into a DataFrame.
        """
        df = load_data('/Users/ekanshtrivedi/Fetch-MLE/data_daily.csv')
        self.assertIsInstance(df, pd.DataFrame)

    def test_feature_engineering(self):
        """
        Test whether the feature_engineering function correctly adds new features to the DataFrame.
        """
        df = pd.DataFrame({'# Date': pd.date_range(start='2021-01-01', periods=10, freq='D'), 
                           'Receipt_Count': range(10)})
        df = feature_engineering(df)
        self.assertIn('month', df.columns)
        self.assertIn('day', df.columns)

    def test_scale_data(self):
        """
        Test whether the scale_data function correctly scales the data.
        """
        df = pd.DataFrame({'Receipt_Count': range(10)})
        train, test, scaler = scale_data(df, df)
        self.assertEqual(train.shape, (10, 1))
        self.assertEqual(test.shape, (10, 1))

    def test_create_sequences(self):
        """
        Test whether the create_sequences function correctly creates sequences from the data.
        """
        data = pd.DataFrame({'Receipt_Count': range(100)}).values
        X, y = create_sequences(data, 10)
        self.assertEqual(X.shape[0], y.shape[0])
        self.assertEqual(X.shape[1], 10)

    def test_split_data(self):
        """
        Test whether the split_data function correctly splits the data into training and testing sets.
        """
        df = pd.DataFrame({'# Date': pd.date_range(start='2021-01-01', periods=10, freq='D'), 
                           'Receipt_Count': range(10)})
        train, test = split_data(feature_engineering(df), '2021-01-06')
        self.assertTrue(train['# Date'].max() < pd.Timestamp('2021-01-06'))
        self.assertTrue(test['# Date'].min() >= pd.Timestamp('2021-01-06'))

    def test_prepare_features(self):
        """
        Test whether the prepare_features function correctly prepares the features and target variables for training and testing.
        """
        df = pd.DataFrame({'# Date': pd.date_range(start='2021-01-01', periods=10, freq='D'), 
                           'Receipt_Count': range(10)})
        train, test = split_data(feature_engineering(df), '2021-01-06')
        X_train, y_train, X_test, y_test = prepare_features(train, test)
        self.assertEqual(X_train.shape[0], y_train.shape[0])
        self.assertEqual(X_test.shape[0], y_test.shape[0])

if __name__ == '__main__':
    unittest.main()