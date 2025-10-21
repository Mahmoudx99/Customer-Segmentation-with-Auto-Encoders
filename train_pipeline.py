"""
Main training pipeline for customer segmentation with Auto Encoders.

This script:
1. Loads and preprocesses data
2. Trains an Auto Encoder
3. Performs clustering on encoded features
4. Evaluates and visualizes results
5. Saves models and predictions
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Add src to path
sys.path.append('src')

from data_loader import load_data
from preprocessing import prepare_data_for_training, CustomerDataPreprocessor
from autoencoder import CustomerAutoEncoder
from clustering import (CustomerClusterer, visualize_clusters_2d,
                       visualize_cluster_distribution, analyze_cluster_characteristics,
                       find_optimal_clusters, map_clusters_to_segments)


def main():
    """
    Main training pipeline.
    """
    print("="*80)
    print("CUSTOMER SEGMENTATION WITH AUTO ENCODERS")
    print("="*80)

    # Configuration
    RANDOM_STATE = 42
    ENCODING_DIM = 8
    HIDDEN_LAYERS = [16, 12]
    N_CLUSTERS = 7
    EPOCHS = 100
    BATCH_SIZE = 32
    VALIDATION_SPLIT = 0.2

    # Create output directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    os.makedirs('results/plots', exist_ok=True)

    # =========================================================================
    # Step 1: Load Data
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 1: Loading Data")
    print("="*80)

    train_df, test_df = load_data('data/raw')

    print(f"\nTarget distribution:")
    print(train_df['Segmentation'].value_counts().sort_index())

    # =========================================================================
    # Step 2: Preprocess Data
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 2: Preprocessing Data")
    print("="*80)

    X_train_full, y_train_full, X_test, test_ids, preprocessor = prepare_data_for_training(
        train_df, test_df,
        save_preprocessor=True,
        preprocessor_path='models/preprocessor.pkl'
    )

    # Split training data for validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full,
        test_size=VALIDATION_SPLIT,
        random_state=RANDOM_STATE,
        stratify=y_train_full
    )

    print(f"\nFinal split:")
    print(f"  Training samples: {X_train.shape[0]}")
    print(f"  Validation samples: {X_val.shape[0]}")
    print(f"  Test samples: {X_test.shape[0]}")

    # =========================================================================
    # Step 3: Build and Train Auto Encoder
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 3: Building and Training Auto Encoder")
    print("="*80)

    input_dim = X_train.shape[1]
    print(f"\nAuto Encoder configuration:")
    print(f"  Input dimension: {input_dim}")
    print(f"  Hidden layers: {HIDDEN_LAYERS}")
    print(f"  Encoding dimension: {ENCODING_DIM}")

    autoencoder = CustomerAutoEncoder(
        input_dim=input_dim,
        encoding_dim=ENCODING_DIM,
        hidden_layers=HIDDEN_LAYERS
    )

    autoencoder.summary()

    print(f"\nTraining Auto Encoder...")
    history = autoencoder.train(
        X_train, X_val,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1
    )

    # Plot training history
    autoencoder.plot_training_history(save_path='results/plots/training_history.png')

    # Save autoencoder
    autoencoder.save('models/autoencoder.h5')

    # =========================================================================
    # Step 4: Encode Features
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 4: Encoding Features")
    print("="*80)

    X_train_full_encoded = autoencoder.encode(X_train_full)
    X_test_encoded = autoencoder.encode(X_test)

    print(f"\nEncoded feature dimensions:")
    print(f"  Training: {X_train_full_encoded.shape}")
    print(f"  Test: {X_test_encoded.shape}")

    # Check reconstruction error
    train_reconstruction_error = autoencoder.get_reconstruction_error(X_train_full)
    print(f"\nReconstruction error (MSE):")
    print(f"  Mean: {train_reconstruction_error.mean():.6f}")
    print(f"  Std: {train_reconstruction_error.std():.6f}")
    print(f"  Min: {train_reconstruction_error.min():.6f}")
    print(f"  Max: {train_reconstruction_error.max():.6f}")

    # =========================================================================
    # Step 5: Find Optimal Number of Clusters (Optional Analysis)
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 5: Analyzing Optimal Number of Clusters")
    print("="*80)

    cluster_analysis = find_optimal_clusters(X_train_full_encoded, max_clusters=12)
    plt.savefig('results/plots/optimal_clusters.png', dpi=300, bbox_inches='tight')

    # =========================================================================
    # Step 6: Perform Clustering
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 6: Performing K-Means Clustering")
    print("="*80)

    clusterer = CustomerClusterer(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE)
    train_cluster_labels = clusterer.fit_predict(X_train_full_encoded)

    # Evaluate clustering
    metrics = clusterer.evaluate(X_train_full_encoded, train_cluster_labels)
    print(f"\nClustering Evaluation Metrics:")
    print(f"  Silhouette Score: {metrics['silhouette_score']:.4f}")
    print(f"  Davies-Bouldin Score: {metrics['davies_bouldin_score']:.4f}")
    print(f"  Calinski-Harabasz Score: {metrics['calinski_harabasz_score']:.4f}")
    print(f"  Inertia: {metrics['inertia']:.4f}")

    # Save clusterer
    clusterer.save('models/clusterer.pkl')

    # =========================================================================
    # Step 7: Visualize Clusters
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 7: Visualizing Clusters")
    print("="*80)

    # t-SNE visualization
    print("\nCreating t-SNE visualization...")
    visualize_clusters_2d(
        X_train_full_encoded, train_cluster_labels,
        method='tsne',
        title='Customer Segments',
        save_path='results/plots/clusters_tsne.png',
        true_labels=y_train_full
    )

    # PCA visualization
    print("\nCreating PCA visualization...")
    visualize_clusters_2d(
        X_train_full_encoded, train_cluster_labels,
        method='pca',
        title='Customer Segments',
        save_path='results/plots/clusters_pca.png',
        true_labels=y_train_full
    )

    # Cluster distribution
    print("\nVisualizing cluster distribution...")
    segment_names = [f'Segment {i}' for i in range(N_CLUSTERS)]
    visualize_cluster_distribution(
        train_cluster_labels,
        segment_names=segment_names,
        save_path='results/plots/cluster_distribution.png'
    )

    # =========================================================================
    # Step 8: Analyze Cluster Characteristics
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 8: Analyzing Cluster Characteristics")
    print("="*80)

    cluster_characteristics = analyze_cluster_characteristics(
        X_train_full,
        train_cluster_labels,
        feature_names=preprocessor.feature_names,
        numerical_indices=[0, 1, 2]  # Age, Work_Experience, Family_Size
    )

    # Save cluster characteristics
    cluster_characteristics.to_csv('results/cluster_characteristics.csv')
    print(f"\nCluster characteristics saved to results/cluster_characteristics.csv")

    # =========================================================================
    # Step 9: Map Clusters to Original Segments
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 9: Mapping Clusters to Original Segments")
    print("="*80)

    cluster_to_segment_map = map_clusters_to_segments(train_cluster_labels, y_train_full)
    print("\nCluster to Segment Mapping:")
    for cluster, segment in sorted(cluster_to_segment_map.items()):
        count = np.sum(train_cluster_labels == cluster)
        print(f"  Cluster {cluster} -> Segment {segment} ({count} samples)")

    # =========================================================================
    # Step 10: Generate Test Predictions
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 10: Generating Test Predictions")
    print("="*80)

    test_cluster_labels = clusterer.predict(X_test_encoded)

    # Map clusters to segments
    test_segment_predictions = np.array([cluster_to_segment_map[cluster]
                                        for cluster in test_cluster_labels])

    print(f"\nTest predictions generated for {len(test_segment_predictions)} samples")
    print(f"Predicted segment distribution:")
    unique, counts = np.unique(test_segment_predictions, return_counts=True)
    for seg, count in zip(unique, counts):
        print(f"  {seg}: {count} ({count/len(test_segment_predictions)*100:.2f}%)")

    # =========================================================================
    # Step 11: Save Results
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 11: Saving Results")
    print("="*80)

    # Save submission file
    submission_df = pd.DataFrame({
        'ID': test_ids,
        'Segmentation': test_segment_predictions
    })
    submission_df.to_csv('results/submission.csv', index=False)
    print(f"Submission file saved to: results/submission.csv")

    # Save detailed results
    detailed_results = pd.DataFrame({
        'ID': test_ids,
        'Cluster': test_cluster_labels,
        'Predicted_Segment': test_segment_predictions
    })
    detailed_results.to_csv('results/test_predictions_detailed.csv', index=False)
    print(f"Detailed predictions saved to: results/test_predictions_detailed.csv")

    # Save training results
    training_results = pd.DataFrame({
        'ID': train_df['ID'].values,
        'True_Segment': y_train_full,
        'Cluster': train_cluster_labels,
        'Predicted_Segment': [cluster_to_segment_map[cluster] for cluster in train_cluster_labels]
    })
    training_results.to_csv('results/training_results.csv', index=False)
    print(f"Training results saved to: results/training_results.csv")

    # Save metrics
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv('results/clustering_metrics.csv', index=False)
    print(f"Clustering metrics saved to: results/clustering_metrics.csv")

    # =========================================================================
    # Final Summary
    # =========================================================================
    print("\n" + "="*80)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("="*80)
    print(f"\nModels saved:")
    print(f"  - models/autoencoder.h5")
    print(f"  - models/preprocessor.pkl")
    print(f"  - models/clusterer.pkl")
    print(f"\nResults saved:")
    print(f"  - results/submission.csv")
    print(f"  - results/test_predictions_detailed.csv")
    print(f"  - results/training_results.csv")
    print(f"  - results/cluster_characteristics.csv")
    print(f"  - results/clustering_metrics.csv")
    print(f"\nPlots saved:")
    print(f"  - results/plots/training_history.png")
    print(f"  - results/plots/optimal_clusters.png")
    print(f"  - results/plots/clusters_tsne.png")
    print(f"  - results/plots/clusters_pca.png")
    print(f"  - results/plots/cluster_distribution.png")
    print("="*80)


if __name__ == "__main__":
    main()
