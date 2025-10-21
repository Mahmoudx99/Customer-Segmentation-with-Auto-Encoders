"""
Data loading utilities for customer segmentation project.
"""
import pandas as pd
import os


def load_data(data_dir='data/raw'):
    """
    Load training and test datasets.

    Args:
        data_dir (str): Directory containing the CSV files

    Returns:
        tuple: (train_df, test_df)
    """
    train_path = os.path.join(data_dir, 'Train.csv')
    test_path = os.path.join(data_dir, 'Test.csv')

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Train.csv not found in {data_dir}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test.csv not found in {data_dir}")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    print(f"Training data shape: {train_df.shape}")
    print(f"Test data shape: {test_df.shape}")

    return train_df, test_df


def get_feature_info(df):
    """
    Get information about features in the dataset.

    Args:
        df (pd.DataFrame): Input dataframe

    Returns:
        dict: Feature information
    """
    info = {
        'shape': df.shape,
        'columns': df.columns.tolist(),
        'dtypes': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'numeric_features': df.select_dtypes(include=['int64', 'float64']).columns.tolist(),
        'categorical_features': df.select_dtypes(include=['object']).columns.tolist()
    }

    return info
