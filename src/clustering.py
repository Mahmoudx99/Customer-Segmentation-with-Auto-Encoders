"""
Clustering utilities for customer segmentation.
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os


class CustomerClusterer:
    """
    K-Means clustering for customer segmentation.
    """

    def __init__(self, n_clusters=7, random_state=42):
        """
        Initialize the clusterer.

        Args:
            n_clusters (int): Number of clusters
            random_state (int): Random state for reproducibility
        """
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init=50,
            max_iter=500
        )
        self.labels_ = None
        self.cluster_centers_ = None

    def fit(self, X):
        """
        Fit the clusterer on encoded features.

        Args:
            X (np.ndarray): Encoded features

        Returns:
            self
        """
        self.kmeans.fit(X)
        self.labels_ = self.kmeans.labels_
        self.cluster_centers_ = self.kmeans.cluster_centers_
        return self

    def predict(self, X):
        """
        Predict cluster labels for new data.

        Args:
            X (np.ndarray): Encoded features

        Returns:
            np.ndarray: Cluster labels
        """
        return self.kmeans.predict(X)

    def fit_predict(self, X):
        """
        Fit and predict in one step.

        Args:
            X (np.ndarray): Encoded features

        Returns:
            np.ndarray: Cluster labels
        """
        return self.kmeans.fit_predict(X)

    def evaluate(self, X, labels=None):
        """
        Evaluate clustering quality.

        Args:
            X (np.ndarray): Encoded features
            labels (np.ndarray): Cluster labels (optional)

        Returns:
            dict: Evaluation metrics
        """
        if labels is None:
            labels = self.labels_

        if labels is None:
            raise ValueError("No labels available. Fit the model first or provide labels.")

        metrics = {
            'silhouette_score': silhouette_score(X, labels),
            'davies_bouldin_score': davies_bouldin_score(X, labels),
            'calinski_harabasz_score': calinski_harabasz_score(X, labels),
            'inertia': self.kmeans.inertia_
        }

        return metrics

    def save(self, filepath):
        """
        Save the clusterer to disk.

        Args:
            filepath (str): Path to save the clusterer
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self, filepath)
        print(f"Clusterer saved to {filepath}")

    @staticmethod
    def load(filepath):
        """
        Load clusterer from disk.

        Args:
            filepath (str): Path to load the clusterer from

        Returns:
            CustomerClusterer: Loaded clusterer
        """
        return joblib.load(filepath)


def map_clusters_to_segments(cluster_labels, true_segments):
    """
    Map cluster labels to segment labels based on majority voting.

    Args:
        cluster_labels (np.ndarray): Predicted cluster labels
        true_segments (np.ndarray): True segment labels

    Returns:
        dict: Mapping from cluster to segment
    """
    unique_clusters = np.unique(cluster_labels)
    cluster_to_segment = {}

    for cluster in unique_clusters:
        mask = cluster_labels == cluster
        segments_in_cluster = true_segments[mask]
        # Find the most common segment in this cluster
        unique, counts = np.unique(segments_in_cluster, return_counts=True)
        most_common_segment = unique[np.argmax(counts)]
        cluster_to_segment[cluster] = most_common_segment

    return cluster_to_segment


def visualize_clusters_2d(X_encoded, labels, method='tsne', title='Customer Segments',
                           save_path=None, true_labels=None):
    """
    Visualize clusters in 2D using dimensionality reduction.

    Args:
        X_encoded (np.ndarray): Encoded features
        labels (np.ndarray): Cluster labels
        method (str): Dimensionality reduction method ('tsne' or 'pca')
        title (str): Plot title
        save_path (str): Path to save the plot (optional)
        true_labels (np.ndarray): True labels for comparison (optional)
    """
    # Dimensionality reduction
    if method.lower() == 'tsne':
        reducer = TSNE(n_components=2, random_state=42, perplexity=30)
        X_2d = reducer.fit_transform(X_encoded)
        method_name = 't-SNE'
    elif method.lower() == 'pca':
        reducer = PCA(n_components=2, random_state=42)
        X_2d = reducer.fit_transform(X_encoded)
        method_name = 'PCA'
        explained_var = reducer.explained_variance_ratio_
        print(f"Explained variance ratio: {explained_var}")
    else:
        raise ValueError("Method must be 'tsne' or 'pca'")

    # Create figure
    n_plots = 2 if true_labels is not None else 1
    fig, axes = plt.subplots(1, n_plots, figsize=(8*n_plots, 6))

    if n_plots == 1:
        axes = [axes]

    # Plot predicted clusters
    scatter = axes[0].scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap='tab10',
                             alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
    axes[0].set_title(f'{title} - Predicted Clusters ({method_name})',
                     fontsize=14, fontweight='bold')
    axes[0].set_xlabel(f'{method_name} Component 1')
    axes[0].set_ylabel(f'{method_name} Component 2')
    plt.colorbar(scatter, ax=axes[0], label='Cluster')
    axes[0].grid(True, alpha=0.3)

    # Plot true labels if provided
    if true_labels is not None:
        # Encode true labels if they're strings
        if true_labels.dtype == object:
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            true_labels_encoded = le.fit_transform(true_labels)
        else:
            true_labels_encoded = true_labels

        scatter2 = axes[1].scatter(X_2d[:, 0], X_2d[:, 1], c=true_labels_encoded,
                                  cmap='tab10', alpha=0.6, s=50,
                                  edgecolors='black', linewidth=0.5)
        axes[1].set_title(f'{title} - True Segments ({method_name})',
                         fontsize=14, fontweight='bold')
        axes[1].set_xlabel(f'{method_name} Component 1')
        axes[1].set_ylabel(f'{method_name} Component 2')
        plt.colorbar(scatter2, ax=axes[1], label='True Segment')
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Cluster visualization saved to {save_path}")

    plt.show()


def visualize_cluster_distribution(labels, segment_names=None, save_path=None):
    """
    Visualize the distribution of clusters.

    Args:
        labels (np.ndarray): Cluster labels
        segment_names (list): Names for segments (optional)
        save_path (str): Path to save the plot (optional)
    """
    unique_labels, counts = np.unique(labels, return_counts=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Bar plot
    if segment_names:
        x_labels = [segment_names[i] if i < len(segment_names) else f'Cluster {i}'
                   for i in unique_labels]
    else:
        x_labels = [f'Cluster {i}' for i in unique_labels]

    axes[0].bar(range(len(unique_labels)), counts, color='steelblue',
               edgecolor='black', alpha=0.7)
    axes[0].set_xticks(range(len(unique_labels)))
    axes[0].set_xticklabels(x_labels, rotation=45, ha='right')
    axes[0].set_title('Cluster Distribution', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Cluster')
    axes[0].set_ylabel('Count')
    axes[0].grid(axis='y', alpha=0.3)

    # Pie chart
    colors = plt.cm.Set3(range(len(unique_labels)))
    axes[1].pie(counts, labels=x_labels, autopct='%1.1f%%', colors=colors,
               startangle=90)
    axes[1].set_title('Cluster Proportions', fontsize=14, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Distribution plot saved to {save_path}")

    plt.show()


def analyze_cluster_characteristics(X_original, labels, feature_names,
                                    numerical_indices=None):
    """
    Analyze characteristics of each cluster.

    Args:
        X_original (np.ndarray): Original (preprocessed) features
        labels (np.ndarray): Cluster labels
        feature_names (list): Feature names
        numerical_indices (list): Indices of numerical features

    Returns:
        pd.DataFrame: Cluster characteristics
    """
    df = pd.DataFrame(X_original, columns=feature_names)
    df['Cluster'] = labels

    # Calculate mean for each cluster
    cluster_means = df.groupby('Cluster').mean()

    print("="*80)
    print("CLUSTER CHARACTERISTICS (Mean Values)")
    print("="*80)
    print(cluster_means)
    print("="*80)

    # Visualize if numerical features are specified
    if numerical_indices is not None and len(numerical_indices) > 0:
        numerical_features = [feature_names[i] for i in numerical_indices]
        cluster_means_numerical = cluster_means[numerical_features]

        fig, ax = plt.subplots(figsize=(12, 6))
        cluster_means_numerical.T.plot(kind='bar', ax=ax, rot=45)
        ax.set_title('Average Feature Values by Cluster (Numerical Features)',
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Feature')
        ax.set_ylabel('Average Value (Scaled)')
        ax.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.show()

    return cluster_means


def find_optimal_clusters(X_encoded, max_clusters=15):
    """
    Find optimal number of clusters using elbow method and silhouette score.

    Args:
        X_encoded (np.ndarray): Encoded features
        max_clusters (int): Maximum number of clusters to try

    Returns:
        dict: Results for different numbers of clusters
    """
    inertias = []
    silhouette_scores = []
    K_range = range(2, max_clusters + 1)

    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=20)
        labels = kmeans.fit_predict(X_encoded)
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X_encoded, labels))

    # Plot results
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Elbow plot
    axes[0].plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
    axes[0].set_title('Elbow Method', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Number of Clusters')
    axes[0].set_ylabel('Inertia')
    axes[0].grid(True, alpha=0.3)
    axes[0].axvline(x=7, color='red', linestyle='--', label='k=7 (Target)')
    axes[0].legend()

    # Silhouette score plot
    axes[1].plot(K_range, silhouette_scores, 'go-', linewidth=2, markersize=8)
    axes[1].set_title('Silhouette Score', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Number of Clusters')
    axes[1].set_ylabel('Silhouette Score')
    axes[1].grid(True, alpha=0.3)
    axes[1].axvline(x=7, color='red', linestyle='--', label='k=7 (Target)')
    axes[1].legend()

    plt.tight_layout()
    plt.show()

    results = {
        'k_values': list(K_range),
        'inertias': inertias,
        'silhouette_scores': silhouette_scores,
        'optimal_k_silhouette': K_range[np.argmax(silhouette_scores)]
    }

    print(f"Optimal k based on silhouette score: {results['optimal_k_silhouette']}")

    return results
