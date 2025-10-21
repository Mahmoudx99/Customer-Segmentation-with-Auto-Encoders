"""
Inference script for making predictions on new customer data.

Usage:
    python predict.py --input data/raw/Test.csv --output results/predictions.csv
"""
import argparse
import sys
import pandas as pd
import numpy as np

sys.path.append('src')

from preprocessing import CustomerDataPreprocessor
from autoencoder import CustomerAutoEncoder
from clustering import CustomerClusterer


def load_models(autoencoder_path='models/autoencoder.h5',
                preprocessor_path='models/preprocessor.pkl',
                clusterer_path='models/clusterer.pkl'):
    """
    Load trained models.

    Args:
        autoencoder_path (str): Path to saved autoencoder
        preprocessor_path (str): Path to saved preprocessor
        clusterer_path (str): Path to saved clusterer

    Returns:
        tuple: (autoencoder, preprocessor, clusterer)
    """
    print("Loading models...")
    autoencoder = CustomerAutoEncoder.load(autoencoder_path)
    preprocessor = CustomerDataPreprocessor.load(preprocessor_path)
    clusterer = CustomerClusterer.load(clusterer_path)
    print("Models loaded successfully!")

    return autoencoder, preprocessor, clusterer


def predict_segments(data_df, autoencoder, preprocessor, clusterer,
                     cluster_to_segment_map=None):
    """
    Predict segments for new customer data.

    Args:
        data_df (pd.DataFrame): Input customer data
        autoencoder (CustomerAutoEncoder): Trained autoencoder
        preprocessor (CustomerDataPreprocessor): Fitted preprocessor
        clusterer (CustomerClusterer): Trained clusterer
        cluster_to_segment_map (dict): Mapping from clusters to segments

    Returns:
        tuple: (predicted_segments, cluster_labels, encoded_features)
    """
    # Extract IDs
    ids = data_df['ID'].values

    # Remove ID column for preprocessing
    X_df = data_df.drop(['ID'], axis=1)

    # Preprocess
    print("Preprocessing data...")
    X_preprocessed = preprocessor.transform(X_df)

    # Encode with autoencoder
    print("Encoding features...")
    X_encoded = autoencoder.encode(X_preprocessed)

    # Predict clusters
    print("Predicting clusters...")
    cluster_labels = clusterer.predict(X_encoded)

    # Map to segments if mapping provided
    if cluster_to_segment_map is not None:
        predicted_segments = np.array([cluster_to_segment_map[cluster]
                                      for cluster in cluster_labels])
    else:
        predicted_segments = cluster_labels

    return predicted_segments, cluster_labels, X_encoded


def main():
    """
    Main inference function.
    """
    parser = argparse.ArgumentParser(
        description='Predict customer segments using trained models'
    )
    parser.add_argument('--input', type=str, default='data/raw/Test.csv',
                       help='Path to input CSV file')
    parser.add_argument('--output', type=str, default='results/predictions.csv',
                       help='Path to save predictions')
    parser.add_argument('--autoencoder', type=str, default='models/autoencoder.h5',
                       help='Path to trained autoencoder')
    parser.add_argument('--preprocessor', type=str, default='models/preprocessor.pkl',
                       help='Path to fitted preprocessor')
    parser.add_argument('--clusterer', type=str, default='models/clusterer.pkl',
                       help='Path to trained clusterer')

    args = parser.parse_args()

    print("="*80)
    print("CUSTOMER SEGMENTATION - INFERENCE")
    print("="*80)

    # Load data
    print(f"\nLoading data from: {args.input}")
    data_df = pd.read_csv(args.input)
    print(f"Loaded {len(data_df)} samples")

    # Load models
    autoencoder, preprocessor, clusterer = load_models(
        args.autoencoder, args.preprocessor, args.clusterer
    )

    # Make predictions
    print("\nMaking predictions...")
    predicted_segments, cluster_labels, X_encoded = predict_segments(
        data_df, autoencoder, preprocessor, clusterer
    )

    # Create output dataframe
    results_df = pd.DataFrame({
        'ID': data_df['ID'].values,
        'Predicted_Segment': predicted_segments,
        'Cluster': cluster_labels
    })

    # Save results
    print(f"\nSaving predictions to: {args.output}")
    results_df.to_csv(args.output, index=False)

    # Print summary
    print("\n" + "="*80)
    print("PREDICTION SUMMARY")
    print("="*80)
    print(f"Total samples: {len(results_df)}")
    print(f"\nPredicted segment distribution:")
    for segment, count in results_df['Predicted_Segment'].value_counts().sort_index().items():
        percentage = (count / len(results_df)) * 100
        print(f"  {segment}: {count} ({percentage:.2f}%)")
    print("="*80)

    print(f"\nPredictions saved successfully to: {args.output}")


if __name__ == "__main__":
    main()
