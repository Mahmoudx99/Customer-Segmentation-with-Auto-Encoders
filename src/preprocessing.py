"""
Data preprocessing utilities for customer segmentation.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import joblib
import os


class CustomerDataPreprocessor:
    """
    Preprocessor for customer segmentation data.
    Handles missing values, encoding, and scaling.
    """

    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.numerical_features = ['Age', 'Work_Experience', 'Family_Size']
        self.categorical_features = ['Gender', 'Ever_Married', 'Graduated',
                                     'Profession', 'Spending_Score', 'Var_1']
        self.feature_names = None
        self.imputer_numerical = SimpleImputer(strategy='median')
        self.imputer_categorical = {}  # Dictionary to store imputers for each categorical feature

    def fit(self, df):
        """
        Fit the preprocessor on training data.

        Args:
            df (pd.DataFrame): Training dataframe

        Returns:
            self
        """
        # Handle missing values first
        df_processed = df.copy()

        # Impute numerical features
        if any(df_processed[self.numerical_features].isnull().any()):
            df_processed[self.numerical_features] = self.imputer_numerical.fit_transform(
                df_processed[self.numerical_features]
            )

        # Impute categorical features
        for col in self.categorical_features:
            imputer = SimpleImputer(strategy='most_frequent')
            df_processed[col] = imputer.fit_transform(
                df_processed[[col]]
            ).ravel()
            self.imputer_categorical[col] = imputer

        # Fit label encoders for categorical features
        for col in self.categorical_features:
            le = LabelEncoder()
            le.fit(df_processed[col].astype(str))
            self.label_encoders[col] = le

        # Prepare data for scaling
        X = self._encode_features(df_processed)

        # Fit scaler
        self.scaler.fit(X)

        # Store feature names
        self.feature_names = (self.numerical_features +
                            [f"{col}_encoded" for col in self.categorical_features])

        return self

    def transform(self, df):
        """
        Transform data using fitted preprocessor.

        Args:
            df (pd.DataFrame): Dataframe to transform

        Returns:
            np.ndarray: Scaled and encoded features
        """
        df_processed = df.copy()

        # Impute numerical features
        df_processed[self.numerical_features] = self.imputer_numerical.transform(
            df_processed[self.numerical_features]
        )

        # Impute categorical features
        for col in self.categorical_features:
            df_processed[col] = self.imputer_categorical[col].transform(
                df_processed[[col]]
            ).ravel()

        # Encode categorical features
        X = self._encode_features(df_processed)

        # Scale features
        X_scaled = self.scaler.transform(X)

        return X_scaled

    def fit_transform(self, df):
        """
        Fit and transform in one step.

        Args:
            df (pd.DataFrame): Dataframe to fit and transform

        Returns:
            np.ndarray: Scaled and encoded features
        """
        return self.fit(df).transform(df)

    def _encode_features(self, df):
        """
        Encode categorical features using label encoding.

        Args:
            df (pd.DataFrame): Dataframe with features

        Returns:
            np.ndarray: Encoded features
        """
        # Get numerical features
        X_numerical = df[self.numerical_features].values

        # Encode categorical features
        X_categorical = []
        for col in self.categorical_features:
            encoded = self.label_encoders[col].transform(df[col].astype(str))
            X_categorical.append(encoded.reshape(-1, 1))

        X_categorical = np.hstack(X_categorical)

        # Combine numerical and categorical
        X = np.hstack([X_numerical, X_categorical])

        return X

    def save(self, filepath):
        """
        Save preprocessor to disk.

        Args:
            filepath (str): Path to save the preprocessor
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self, filepath)
        print(f"Preprocessor saved to {filepath}")

    @staticmethod
    def load(filepath):
        """
        Load preprocessor from disk.

        Args:
            filepath (str): Path to load the preprocessor from

        Returns:
            CustomerDataPreprocessor: Loaded preprocessor
        """
        return joblib.load(filepath)


def prepare_data_for_training(train_df, test_df, save_preprocessor=True,
                               preprocessor_path='models/preprocessor.pkl'):
    """
    Prepare training and test data for model training.

    Args:
        train_df (pd.DataFrame): Training dataframe
        test_df (pd.DataFrame): Test dataframe
        save_preprocessor (bool): Whether to save the preprocessor
        preprocessor_path (str): Path to save the preprocessor

    Returns:
        tuple: (X_train, y_train, X_test, test_ids, preprocessor)
    """
    # Remove ID column and extract labels
    train_ids = train_df['ID'].values
    test_ids = test_df['ID'].values

    # Extract target variable
    y_train = train_df['Segmentation'].values

    # Remove ID and target from features
    X_train_df = train_df.drop(['ID', 'Segmentation'], axis=1)
    X_test_df = test_df.drop(['ID'], axis=1)

    # Initialize and fit preprocessor
    preprocessor = CustomerDataPreprocessor()
    X_train = preprocessor.fit_transform(X_train_df)
    X_test = preprocessor.transform(X_test_df)

    # Save preprocessor
    if save_preprocessor:
        preprocessor.save(preprocessor_path)

    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    print(f"Number of features: {X_train.shape[1]}")
    print(f"Number of training samples: {X_train.shape[0]}")
    print(f"Number of test samples: {X_test.shape[0]}")

    return X_train, y_train, X_test, test_ids, preprocessor
