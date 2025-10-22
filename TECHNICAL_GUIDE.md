# Customer Segmentation with Auto Encoders - Complete Technical Guide

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Problem Statement](#2-problem-statement)
3. [Dataset Description](#3-dataset-description)
4. [Theoretical Background](#4-theoretical-background)
5. [Technical Architecture](#5-technical-architecture)
6. [Implementation Details](#6-implementation-details)
7. [Pipeline Workflow](#7-pipeline-workflow)
8. [Evaluation Metrics](#8-evaluation-metrics)
9. [Results Interpretation](#9-results-interpretation)
10. [Use Cases and Applications](#10-use-cases-and-applications)
11. [Advanced Topics](#11-advanced-topics)
12. [References](#12-references)

---

## 1. Project Overview

### 1.1 Executive Summary

This project implements an unsupervised machine learning solution for customer segmentation using **Auto Encoders** (a type of neural network) combined with **K-Means clustering**. The goal is to classify customers into 7 distinct segments for an automobile company's market expansion strategy.

### 1.2 Business Context

An automobile company wants to enter new markets with their existing products (P1-P5). Their sales team has successfully used segmented outreach strategies in their existing market, classifying customers into 7 categories. They now want to apply the same segmentation approach to 2,627 potential customers in the new market.

### 1.3 Key Innovation

Instead of using traditional clustering directly on raw features, we use **Auto Encoders** to:
1. Learn compressed, non-linear representations of customer features
2. Reduce dimensionality while preserving important patterns
3. Extract features that better capture customer similarities
4. Improve clustering quality by working in a learned latent space

### 1.4 Project Goals

- **Primary**: Segment 2,627 new customers into 7 meaningful groups
- **Secondary**: Understand customer characteristics within each segment
- **Technical**: Demonstrate the effectiveness of Auto Encoder + clustering approach
- **Business**: Enable targeted marketing strategies for each segment

---

## 2. Problem Statement

### 2.1 The Challenge

**Input**: Customer data with mixed features (numerical and categorical)
- 8,068 training samples with known segments (A, B, C, D)
- 2,627 test samples requiring segment prediction
- 9 features describing customer characteristics

**Output**: Segment assignment (7 categories) for each customer

**Constraints**:
- Must handle missing values
- Must work with mixed data types
- Must identify meaningful patterns
- Must be reproducible and scalable

### 2.2 Why Not Simple Clustering?

Traditional K-Means clustering on raw features has limitations:

1. **Curse of Dimensionality**: 9 features may contain redundant information
2. **Non-linear Relationships**: Linear methods miss complex patterns
3. **Scale Sensitivity**: Features have different ranges and importance
4. **Categorical Data**: K-Means works poorly with encoded categories

### 2.3 Why Auto Encoders?

Auto Encoders address these issues by:

1. **Dimensionality Reduction**: Compress 9 features → 8 latent features
2. **Non-linear Transformation**: Capture complex relationships
3. **Feature Learning**: Automatically discover important patterns
4. **Noise Reduction**: Filter out irrelevant variations

---

## 3. Dataset Description

### 3.1 Data Files

| File | Samples | Features | Purpose |
|------|---------|----------|---------|
| Train.csv | 8,068 | 11 | Training and validation |
| Test.csv | 2,627 | 10 | Prediction target |

**Note**: Training data includes `Segmentation` column (target), test data does not.

### 3.2 Feature Descriptions

#### Numerical Features

1. **Age** (integer)
   - Customer age in years
   - Range: Typically 18-90
   - Distribution: May vary across segments
   - Importance: Key demographic indicator

2. **Work_Experience** (float)
   - Years of professional work experience
   - Range: 0.0 to 40+ years
   - May contain missing values
   - Related to income and life stage

3. **Family_Size** (float)
   - Number of family members (including customer)
   - Range: 1 to 9+
   - May contain missing values
   - Affects purchasing decisions (e.g., vehicle size)

#### Categorical Features

4. **Gender** (string)
   - Values: "Male", "Female"
   - May influence product preferences
   - Binary encoding in preprocessing

5. **Ever_Married** (string)
   - Values: "Yes", "No"
   - Indicates marital status
   - Correlates with family size and purchasing power

6. **Graduated** (string)
   - Values: "Yes", "No"
   - Indicates if customer completed graduation
   - Proxy for education level and income

7. **Profession** (string)
   - Customer's occupation
   - Values: "Healthcare", "Engineer", "Lawyer", "Executive", "Artist", "Doctor", "Homemaker", "Entertainment", "Marketing"
   - Strong indicator of income and lifestyle
   - May contain missing values

8. **Spending_Score** (string)
   - Customer's spending behavior
   - Values: "Low", "Average", "High"
   - Directly relevant to purchasing capacity
   - Key for segmentation

9. **Var_1** (string)
   - Label variable with categories
   - Values: "Cat_1" through "Cat_7"
   - Additional categorical feature
   - May encode survey responses or derived attributes

#### Target Variable

10. **Segmentation** (string) - *Training data only*
    - Values: "A", "B", "C", "D"
    - **Note**: Original description mentions 7 categories (cat_1 to cat_7), but actual data has 4 categories
    - Ground truth for training set
    - Used to evaluate clustering quality

#### Identifier

11. **ID** (integer)
    - Unique customer identifier
    - Not used in modeling
    - Required for submission

### 3.3 Data Quality Issues

1. **Missing Values**:
   - Work_Experience: ~15% missing
   - Family_Size: ~10% missing
   - Profession: ~20% missing
   - Other features may have sporadic missing values

2. **Data Types**:
   - Mixed numerical and categorical
   - Requires different preprocessing strategies

3. **Scale Differences**:
   - Age: 0-100
   - Work_Experience: 0-50
   - Family_Size: 1-10
   - Requires normalization

4. **Categorical Cardinality**:
   - Low: Gender (2 values), Ever_Married (2 values)
   - Medium: Spending_Score (3 values)
   - High: Profession (9+ values), Var_1 (7 values)

---

## 4. Theoretical Background

### 4.1 Auto Encoders

#### What is an Auto Encoder?

An **Auto Encoder** is a neural network trained to:
1. **Compress** input data into a lower-dimensional representation (encoding)
2. **Reconstruct** the original data from this representation (decoding)

```
Input (9 features) → Encoder → Latent Space (8 features) → Decoder → Output (9 features)
```

#### Architecture Components

**1. Encoder**: Compresses input data
```
Input Layer (9 units)
    ↓
Hidden Layer 1 (16 units) + ReLU + Dropout(0.2)
    ↓
Hidden Layer 2 (12 units) + ReLU + Dropout(0.2)
    ↓
Bottleneck Layer (8 units) + ReLU
```

**2. Bottleneck Layer**: The compressed representation
- 8-dimensional latent space
- Captures most important information
- This is what we use for clustering

**3. Decoder**: Reconstructs input data (mirror of encoder)
```
Bottleneck (8 units)
    ↓
Hidden Layer 1 (12 units) + ReLU + Dropout(0.2)
    ↓
Hidden Layer 2 (16 units) + ReLU + Dropout(0.2)
    ↓
Output Layer (9 units) + Linear
```

#### Why This Architecture?

1. **Progressive Compression**: 9 → 16 → 12 → 8
   - Gradual dimension reduction
   - Prevents information loss
   - Allows network to learn hierarchical features

2. **Symmetric Design**:
   - Decoder mirrors encoder
   - Ensures reconstruction capability
   - Balances compression and reconstruction

3. **Dropout Layers** (20%):
   - Prevents overfitting
   - Improves generalization
   - Makes features more robust

4. **ReLU Activation**:
   - Non-linear transformations
   - Captures complex relationships
   - Computationally efficient

5. **Bottleneck Size (8)**:
   - Slightly smaller than input (9)
   - Forces compression
   - But not too aggressive (maintains information)

#### Training Process

**Objective**: Minimize reconstruction error

```
Loss = MSE(Input, Reconstructed_Output)
     = Mean Squared Error between original and reconstructed data
```

**What the Network Learns**:
- Which features are most important
- How features relate to each other
- Efficient representations of customer patterns
- Noise filtering (irrelevant variations)

**Training Details**:
- Optimizer: Adam (adaptive learning rate)
- Initial learning rate: 0.001
- Loss function: Mean Squared Error (MSE)
- Metric: Mean Absolute Error (MAE)
- Early stopping: Patience = 15 epochs
- Learning rate reduction: Factor = 0.5, Patience = 5

#### Key Properties

1. **Unsupervised Learning**:
   - No labels required during training
   - Learns from data structure itself
   - Self-supervised (input = output target)

2. **Non-linear Dimensionality Reduction**:
   - Better than PCA for complex patterns
   - Can capture non-linear relationships
   - Learns optimal compression for the data

3. **Feature Learning**:
   - Automatically discovers useful features
   - No manual feature engineering required
   - Adapts to data distribution

### 4.2 K-Means Clustering

#### What is K-Means?

An algorithm that partitions data into K clusters by:
1. Initializing K cluster centers (centroids)
2. Assigning each point to nearest centroid
3. Updating centroids based on assigned points
4. Repeating until convergence

#### Algorithm Steps

```
1. Initialize: Randomly place 7 centroids in 8D space
2. Assignment: For each customer, find closest centroid
3. Update: Move each centroid to mean of assigned customers
4. Repeat: Steps 2-3 until centroids stop moving
```

#### Mathematical Formulation

**Objective**: Minimize within-cluster variance

```
J = Σᵢ₌₁ᴷ Σₓ∈Cᵢ ||x - μᵢ||²

Where:
- K = 7 (number of clusters)
- Cᵢ = cluster i
- x = customer in latent space
- μᵢ = centroid of cluster i
- ||·|| = Euclidean distance
```

#### Why K-Means?

1. **Simplicity**: Easy to understand and implement
2. **Scalability**: Efficient for large datasets
3. **Interpretability**: Clear cluster assignments
4. **Well-studied**: Proven effectiveness

#### Configuration

```python
KMeans(
    n_clusters=7,           # Target number of segments
    n_init=50,              # Run algorithm 50 times
    max_iter=500,           # Maximum iterations per run
    random_state=42         # Reproducibility
)
```

**n_init=50**: Runs algorithm 50 times with different initializations and picks best result. This addresses K-Means' sensitivity to initialization.

### 4.3 Why Combine Auto Encoder + K-Means?

#### The Synergy

1. **Auto Encoder Benefits**:
   - Learns non-linear features
   - Reduces noise
   - Compresses dimensions
   - Handles mixed data types (after preprocessing)

2. **K-Means Benefits**:
   - Fast and scalable
   - Clear cluster assignments
   - Well-understood behavior
   - Easy to interpret

3. **Combined Benefits**:
   - Better clustering quality (higher silhouette scores)
   - More meaningful segments
   - Captures complex patterns
   - Robust to noise

#### Comparison with Alternatives

| Approach | Pros | Cons |
|----------|------|------|
| **K-Means on Raw Data** | Simple, fast | Poor with high dimensions, linear only |
| **PCA + K-Means** | Linear dimension reduction | Cannot capture non-linear patterns |
| **Auto Encoder + K-Means** | Non-linear, learns features | More complex, requires training |
| **Deep Clustering** | End-to-end | Less interpretable, harder to tune |
| **DBSCAN** | Finds arbitrary shapes | Requires density parameters |
| **Hierarchical** | No need to specify K | Computationally expensive |

---

## 5. Technical Architecture

### 5.1 System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Data Ingestion Layer                         │
├─────────────────────────────────────────────────────────────────┤
│  - Load Train.csv (8,068 samples)                               │
│  - Load Test.csv (2,627 samples)                                │
│  - Validate data integrity                                       │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Preprocessing Layer                             │
├─────────────────────────────────────────────────────────────────┤
│  1. Handle Missing Values                                        │
│     - Numerical: Median imputation                               │
│     - Categorical: Most frequent imputation                      │
│                                                                  │
│  2. Encode Categorical Features                                  │
│     - LabelEncoder for each categorical variable                 │
│     - Maintains consistency train/test                           │
│                                                                  │
│  3. Scale Features                                               │
│     - StandardScaler (mean=0, std=1)                            │
│     - Fitted on training data only                               │
│                                                                  │
│  Output: X_train (8068, 9), X_test (2627, 9)                    │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Auto Encoder Layer                            │
├─────────────────────────────────────────────────────────────────┤
│  Architecture:                                                   │
│  Input(9) → Dense(16) → Dense(12) → Bottleneck(8)              │
│           → Dense(12) → Dense(16) → Output(9)                   │
│                                                                  │
│  Training:                                                       │
│  - Loss: MSE (reconstruction error)                             │
│  - Optimizer: Adam (lr=0.001)                                   │
│  - Callbacks: EarlyStopping, ReduceLROnPlateau                  │
│  - Epochs: Up to 100 (with early stopping)                      │
│                                                                  │
│  Output: Encoded features (8068, 8), (2627, 8)                  │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Clustering Layer                             │
├─────────────────────────────────────────────────────────────────┤
│  Algorithm: K-Means                                              │
│  Parameters:                                                     │
│  - n_clusters: 7                                                │
│  - n_init: 50 (multiple random initializations)                 │
│  - max_iter: 500                                                │
│                                                                  │
│  Process:                                                        │
│  1. Fit on training encoded features                            │
│  2. Find 7 cluster centroids                                    │
│  3. Assign each customer to nearest centroid                    │
│                                                                  │
│  Output: Cluster labels (0-6) for each customer                 │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Evaluation Layer                              │
├─────────────────────────────────────────────────────────────────┤
│  Metrics:                                                        │
│  - Silhouette Score (cluster cohesion/separation)               │
│  - Davies-Bouldin Score (cluster similarity)                    │
│  - Calinski-Harabasz Score (variance ratio)                     │
│  - Inertia (within-cluster sum of squares)                      │
│                                                                  │
│  Visualizations:                                                 │
│  - t-SNE 2D projection                                          │
│  - PCA 2D projection                                            │
│  - Cluster distribution plots                                    │
│  - Feature importance by cluster                                 │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Output Layer                                │
├─────────────────────────────────────────────────────────────────┤
│  1. Trained Models:                                              │
│     - autoencoder.h5 (full model)                               │
│     - preprocessor.pkl (scaler + encoders)                      │
│     - clusterer.pkl (K-Means model)                             │
│                                                                  │
│  2. Predictions:                                                 │
│     - submission.csv (ID, Segmentation)                         │
│     - training_results.csv (with true labels)                   │
│     - test_predictions_detailed.csv                             │
│                                                                  │
│  3. Analysis:                                                    │
│     - cluster_characteristics.csv                                │
│     - clustering_metrics.csv                                     │
│                                                                  │
│  4. Visualizations:                                              │
│     - training_history.png                                       │
│     - clusters_tsne.png                                         │
│     - clusters_pca.png                                          │
│     - cluster_distribution.png                                   │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 Component Interactions

```
┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│ data_loader  │─────▶│preprocessing │─────▶│ autoencoder  │
│              │      │              │      │              │
│ - load_data()│      │ - impute()   │      │ - train()    │
│              │      │ - encode()   │      │ - encode()   │
│              │      │ - scale()    │      │              │
└──────────────┘      └──────────────┘      └──────────────┘
                                                    │
                                                    ▼
┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│ Results/     │◀─────│ Evaluation   │◀─────│  clustering  │
│ Output       │      │              │      │              │
│              │      │ - metrics()  │      │ - fit()      │
│ - CSV files  │      │ - visualize()│      │ - predict()  │
│ - Plots      │      │              │      │              │
└──────────────┘      └──────────────┘      └──────────────┘
```

### 5.3 Data Flow

```
Raw Data (CSV)
    │
    ├─ Train: 8,068 × 11 features
    └─ Test:  2,627 × 10 features
    │
    ▼
[Preprocessing]
    │
    ├─ Impute missing values
    ├─ Encode categorical → numerical
    └─ Scale to mean=0, std=1
    │
    ▼
Preprocessed Data
    │
    ├─ Train: 8,068 × 9 features (normalized)
    └─ Test:  2,627 × 9 features (normalized)
    │
    ▼
[Auto Encoder Training]
    │
    ├─ Learn to reconstruct input
    ├─ Minimize MSE loss
    └─ Extract bottleneck features
    │
    ▼
Encoded Data
    │
    ├─ Train: 8,068 × 8 latent features
    └─ Test:  2,627 × 8 latent features
    │
    ▼
[K-Means Clustering]
    │
    ├─ Fit 7 clusters on training data
    ├─ Assign customers to clusters
    └─ Predict test data clusters
    │
    ▼
Cluster Assignments
    │
    ├─ Train: 8,068 cluster labels (0-6)
    └─ Test:  2,627 cluster labels (0-6)
    │
    ▼
[Evaluation & Mapping]
    │
    ├─ Calculate metrics
    ├─ Map clusters → segments (A, B, C, D)
    └─ Generate visualizations
    │
    ▼
Final Output
    │
    ├─ Submission file: ID, Predicted Segment
    ├─ Visualizations: t-SNE, PCA plots
    └─ Analysis: Cluster characteristics
```

### 5.4 File Structure and Dependencies

```
src/
├── __init__.py
│
├── data_loader.py
│   ├── load_data()                # Load CSV files
│   └── get_feature_info()         # Feature metadata
│
├── preprocessing.py
│   ├── CustomerDataPreprocessor   # Main preprocessing class
│   │   ├── fit()                  # Fit on training data
│   │   ├── transform()            # Apply transformations
│   │   ├── fit_transform()        # Fit and transform
│   │   └── _encode_features()     # Internal encoding
│   └── prepare_data_for_training() # High-level wrapper
│
├── autoencoder.py
│   └── CustomerAutoEncoder        # Auto Encoder class
│       ├── __init__()             # Build architecture
│       ├── _build_model()         # Create TensorFlow model
│       ├── train()                # Training loop
│       ├── encode()               # Get latent features
│       ├── decode()               # Reconstruct from latent
│       ├── reconstruct()          # Full reconstruction
│       └── plot_training_history() # Visualize training
│
└── clustering.py
    ├── CustomerClusterer          # K-Means wrapper
    │   ├── fit()                  # Fit clustering
    │   ├── predict()              # Predict clusters
    │   └── evaluate()             # Calculate metrics
    ├── map_clusters_to_segments() # Cluster → segment mapping
    ├── visualize_clusters_2d()    # t-SNE/PCA visualization
    ├── visualize_cluster_distribution() # Distribution plots
    ├── analyze_cluster_characteristics() # Cluster analysis
    └── find_optimal_clusters()    # Elbow method

Scripts:
├── train_pipeline.py              # Main training script
├── predict.py                     # Inference script
└── download_data.py               # Data download utility

Notebooks:
└── 01_exploratory_data_analysis.ipynb  # EDA notebook
```

---

## 6. Implementation Details

### 6.1 Preprocessing Pipeline

#### Step 1: Missing Value Imputation

**Numerical Features** (Age, Work_Experience, Family_Size):
```python
imputer_numerical = SimpleImputer(strategy='median')
```
- **Why median?** Robust to outliers
- **Example**: If Work_Experience has values [0, 1, 2, NaN, 5, 10], missing value becomes median(0,1,2,5,10) = 2

**Categorical Features** (Gender, Ever_Married, etc.):
```python
imputer_categorical = SimpleImputer(strategy='most_frequent')
```
- **Why most frequent?** Preserves mode, works for categories
- **Example**: If Profession has [Engineer, NaN, Doctor, Engineer], missing value becomes "Engineer"

#### Step 2: Categorical Encoding

**Label Encoding** for each categorical variable:
```python
LabelEncoder().fit_transform(['Male', 'Female', 'Male'])
# Returns: [1, 0, 1]
```

**Why LabelEncoder?**
- Simple numerical representation
- Preserves ordinality where relevant
- Memory efficient
- Compatible with StandardScaler

**Example Encoding**:
```
Gender:
  Male   → 1
  Female → 0

Ever_Married:
  Yes → 1
  No  → 0

Spending_Score:
  Low     → 0
  Average → 1
  High    → 2
```

#### Step 3: Feature Scaling

**StandardScaler** (Z-score normalization):
```python
scaled_value = (value - mean) / std_deviation
```

**Example**:
```
Age values: [22, 38, 67, 45, 28]
Mean: 40, Std: 17

Scaled:
  22 → (22-40)/17 = -1.06
  38 → (38-40)/17 = -0.12
  67 → (67-40)/17 = +1.59
```

**Why StandardScaler?**
- Brings all features to same scale (mean=0, std=1)
- Required for neural networks (gradient stability)
- Improves K-Means performance (distance-based)
- Prevents features with larger ranges from dominating

#### Complete Preprocessing Flow

```python
# Fit on training data
preprocessor = CustomerDataPreprocessor()
X_train = preprocessor.fit_transform(train_df)

# Apply to test data (using training statistics)
X_test = preprocessor.transform(test_df)
```

**Important**: Test data is transformed using training data statistics (mean, std) to prevent data leakage.

### 6.2 Auto Encoder Implementation

#### Network Architecture Details

```python
# Input Layer
input_layer = Input(shape=(9,))

# Encoder
x = Dense(16, activation='relu')(input_layer)
x = Dropout(0.2)(x)
x = Dense(12, activation='relu')(x)
x = Dropout(0.2)(x)

# Bottleneck (Latent Space)
encoded = Dense(8, activation='relu', name='bottleneck')(x)

# Decoder
x = Dense(12, activation='relu')(encoded)
x = Dropout(0.2)(x)
x = Dense(16, activation='relu')(x)
x = Dropout(0.2)(x)

# Output Layer
decoded = Dense(9, activation='linear')(x)

# Create Model
autoencoder = Model(inputs=input_layer, outputs=decoded)
```

#### Layer-by-Layer Explanation

**Layer 1: Input → Dense(16)**
- Input: 9 features
- Output: 16 activations
- Purpose: Initial feature expansion
- Activation: ReLU (f(x) = max(0, x))
- Parameters: 9×16 + 16 = 160

**Layer 2: Dropout(0.2)**
- Randomly sets 20% of activations to zero
- Purpose: Regularization, prevent overfitting
- Only active during training

**Layer 3: Dense(16) → Dense(12)**
- Input: 16 activations
- Output: 12 activations
- Purpose: Begin compression
- Parameters: 16×12 + 12 = 204

**Layer 4: Bottleneck Dense(8)**
- Input: 12 activations
- Output: 8 latent features
- Purpose: Compressed representation
- This is what we use for clustering!
- Parameters: 12×8 + 8 = 104

**Layer 5-8: Decoder (Mirror of Encoder)**
- Reconstructs from latent space back to 9 features
- Same structure but reversed
- Final activation: Linear (no constraint on output range)

**Total Parameters**: ~1,200 trainable parameters

#### Training Configuration

```python
autoencoder.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae']
)
```

**Loss Function: Mean Squared Error (MSE)**
```python
MSE = (1/n) Σ(input_i - reconstructed_i)²
```
- Measures reconstruction quality
- Lower is better
- Penalizes large errors heavily

**Optimizer: Adam**
- Adaptive learning rate
- Momentum-based
- Fast convergence
- Handles sparse gradients well

**Callbacks**:

1. **EarlyStopping**
```python
EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True
)
```
- Stops training if validation loss doesn't improve for 15 epochs
- Prevents overfitting
- Restores weights from best epoch

2. **ReduceLROnPlateau**
```python
ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-6
)
```
- Reduces learning rate if loss plateaus
- Helps escape local minima
- Enables fine-tuning

#### Training Process

```python
history = autoencoder.fit(
    X_train, X_train,  # Input = Output (reconstruction)
    validation_data=(X_val, X_val),
    epochs=100,
    batch_size=32,
    callbacks=[early_stopping, reduce_lr]
)
```

**Batch Processing**:
- Processes 32 samples at a time
- Updates weights after each batch
- More stable than single-sample updates
- Faster than full-batch processing

**Validation Split**:
- 80% training, 20% validation
- Monitors generalization
- Prevents overfitting

#### Feature Extraction

```python
encoder = Model(inputs=input_layer, outputs=bottleneck_layer)
latent_features = encoder.predict(X_train)
```

Result: 8-dimensional representation of each customer

**Example**:
```
Original customer: [22, Male, No, 1.0, ..., Low, Cat_4]
                   (9 features after preprocessing)

Encoded customer:  [0.23, -1.45, 0.67, ..., -0.12]
                   (8 latent features)
```

### 6.3 Clustering Implementation

#### K-Means Configuration

```python
kmeans = KMeans(
    n_clusters=7,        # Target segments
    n_init=50,           # Run 50 times with different initializations
    max_iter=500,        # Maximum iterations per run
    random_state=42      # Reproducibility
)
```

#### Algorithm Execution

**Step 1: Initialization** (50 times)
- Randomly place 7 centroids in 8D latent space
- Uses k-means++ initialization (smart initial placement)

**Step 2: Assignment**
For each customer:
```python
distance_to_each_centroid = [
    euclidean_distance(customer, centroid_0),
    euclidean_distance(customer, centroid_1),
    ...,
    euclidean_distance(customer, centroid_6)
]
assigned_cluster = argmin(distance_to_each_centroid)
```

**Step 3: Update**
For each cluster:
```python
new_centroid = mean_of_all_customers_in_cluster
```

**Step 4: Convergence Check**
- If centroids moved < threshold: converged
- Else: return to Step 2
- Max 500 iterations

**Step 5: Best Run Selection**
- Of the 50 runs, select the one with lowest inertia
- Inertia = sum of squared distances to nearest centroid

#### Cluster Assignment

```python
# Training data
train_labels = kmeans.fit_predict(X_train_encoded)
# Returns: array([3, 1, 5, 2, ...])  # Cluster IDs 0-6

# Test data
test_labels = kmeans.predict(X_test_encoded)
```

#### Mapping Clusters to Segments

Since training data has true segments (A, B, C, D), we can map clusters to segments:

```python
def map_clusters_to_segments(cluster_labels, true_segments):
    for cluster in range(7):
        customers_in_cluster = true_segments[cluster_labels == cluster]
        most_common = mode(customers_in_cluster)
        cluster_to_segment[cluster] = most_common
```

**Example**:
```
Cluster 0: [A, A, B, A, A] → Mapped to "A"
Cluster 1: [C, C, C, D, C] → Mapped to "C"
Cluster 2: [B, B, B, B, A] → Mapped to "B"
...
```

### 6.4 Evaluation Implementation

#### Silhouette Score

Measures how similar a customer is to their own cluster vs. other clusters.

```python
silhouette_score = mean(s(i) for i in all_customers)

where s(i) = (b(i) - a(i)) / max(a(i), b(i))

a(i) = average distance to other customers in same cluster
b(i) = average distance to customers in nearest other cluster
```

**Range**: -1 to +1
- +1: Perfect clustering
- 0: Overlapping clusters
- -1: Wrong clusters

#### Davies-Bouldin Score

Measures average similarity between each cluster and its most similar cluster.

```python
DB = (1/K) Σᵢ max(R_ij)

where R_ij = (s_i + s_j) / d_ij

s_i = average distance within cluster i
d_ij = distance between centroids i and j
```

**Range**: 0 to ∞
- Lower is better
- 0 = perfect separation

#### Calinski-Harabasz Score

Ratio of between-cluster to within-cluster variance.

```python
CH = (SSB / (K-1)) / (SSW / (N-K))

SSB = sum of squared distances between cluster centers and overall center
SSW = sum of squared distances within clusters
K = number of clusters
N = number of samples
```

**Range**: 0 to ∞
- Higher is better
- Higher = more separated clusters

#### Visualization

**t-SNE** (t-Distributed Stochastic Neighbor Embedding):
- Non-linear dimensionality reduction
- 8D → 2D for visualization
- Preserves local structure
- Good for visualizing clusters

**PCA** (Principal Component Analysis):
- Linear dimensionality reduction
- 8D → 2D
- Shows global structure
- Provides explained variance

---

## 7. Pipeline Workflow

### 7.1 Training Pipeline (train_pipeline.py)

#### Complete Step-by-Step Process

**Step 1: Data Loading**
```python
train_df, test_df = load_data('data/raw')
# Output:
#   train_df: 8,068 rows × 11 columns
#   test_df:  2,627 rows × 10 columns
```

**Step 2: Data Preprocessing**
```python
X_train, y_train, X_test, test_ids, preprocessor = prepare_data_for_training(
    train_df, test_df
)
# Output:
#   X_train: (8068, 9) - normalized features
#   y_train: (8068,)   - segment labels
#   X_test:  (2627, 9) - normalized features
#   test_ids: (2627,)  - customer IDs
#   preprocessor: fitted preprocessing object
```

**Step 3: Train/Validation Split**
```python
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full,
    test_size=0.2,
    stratify=y_train_full
)
# Output:
#   X_train: (6454, 9)
#   X_val:   (1614, 9)
```

**Step 4: Build Auto Encoder**
```python
autoencoder = CustomerAutoEncoder(
    input_dim=9,
    encoding_dim=8,
    hidden_layers=[16, 12]
)
# Creates neural network architecture
```

**Step 5: Train Auto Encoder**
```python
history = autoencoder.train(
    X_train, X_val,
    epochs=100,
    batch_size=32
)
# Training progress:
# Epoch 1/100: loss: 1.523, val_loss: 1.412
# Epoch 2/100: loss: 1.234, val_loss: 1.198
# ...
# Epoch 45/100: loss: 0.856, val_loss: 0.891
# Early stopping triggered
```

**Step 6: Encode Features**
```python
X_train_encoded = autoencoder.encode(X_train_full)
X_test_encoded = autoencoder.encode(X_test)
# Output:
#   X_train_encoded: (8068, 8)
#   X_test_encoded:  (2627, 8)
```

**Step 7: Perform Clustering**
```python
clusterer = CustomerClusterer(n_clusters=7)
train_labels = clusterer.fit_predict(X_train_encoded)
# Output: array([3, 1, 5, 2, 3, 0, ...])
```

**Step 8: Evaluate Clustering**
```python
metrics = clusterer.evaluate(X_train_encoded)
# Output:
# {
#     'silhouette_score': 0.2952,
#     'davies_bouldin_score': 1.1055,
#     'calinski_harabasz_score': 1234.56,
#     'inertia': 8976.43
# }
```

**Step 9: Visualize Results**
```python
visualize_clusters_2d(X_train_encoded, train_labels, method='tsne')
visualize_cluster_distribution(train_labels)
# Saves: results/plots/clusters_tsne.png
#        results/plots/cluster_distribution.png
```

**Step 10: Map Clusters to Segments**
```python
cluster_to_segment = map_clusters_to_segments(train_labels, y_train_full)
# Example output:
# {0: 'A', 1: 'C', 2: 'B', 3: 'D', 4: 'A', 5: 'B', 6: 'C'}
```

**Step 11: Generate Predictions**
```python
test_labels = clusterer.predict(X_test_encoded)
test_segments = [cluster_to_segment[c] for c in test_labels]
# Output: ['A', 'B', 'D', 'C', ...]
```

**Step 12: Save Results**
```python
# Models
autoencoder.save('models/autoencoder.h5')
preprocessor.save('models/preprocessor.pkl')
clusterer.save('models/clusterer.pkl')

# Predictions
submission_df.to_csv('results/submission.csv')
# Format: ID, Segmentation
#         462809, A
#         462643, B
#         ...

# Analysis files
cluster_characteristics.to_csv('results/cluster_characteristics.csv')
metrics_df.to_csv('results/clustering_metrics.csv')
```

### 7.2 Inference Pipeline (predict.py)

#### Making Predictions on New Data

**Step 1: Load Trained Models**
```python
autoencoder = CustomerAutoEncoder.load('models/autoencoder.h5')
preprocessor = CustomerDataPreprocessor.load('models/preprocessor.pkl')
clusterer = CustomerClusterer.load('models/clusterer.pkl')
```

**Step 2: Load New Data**
```python
new_data = pd.read_csv('data/new_customers.csv')
```

**Step 3: Preprocess**
```python
X_new = preprocessor.transform(new_data)
```

**Step 4: Encode**
```python
X_new_encoded = autoencoder.encode(X_new)
```

**Step 5: Predict Clusters**
```python
cluster_labels = clusterer.predict(X_new_encoded)
```

**Step 6: Map to Segments**
```python
segment_labels = [cluster_to_segment[c] for c in cluster_labels]
```

**Step 7: Save Predictions**
```python
results = pd.DataFrame({
    'ID': new_data['ID'],
    'Predicted_Segment': segment_labels
})
results.to_csv('results/new_predictions.csv')
```

### 7.3 Exploratory Data Analysis (EDA)

The Jupyter notebook provides comprehensive analysis:

**1. Data Loading and Overview**
- Dataset shapes and sizes
- Feature types and dtypes
- First inspection of data

**2. Missing Value Analysis**
- Count and percentage of missing values per feature
- Visualization of missing data patterns
- Impact assessment

**3. Target Variable Analysis**
- Distribution of segments (A, B, C, D)
- Class balance check
- Pie charts and bar plots

**4. Numerical Feature Analysis**
- Age distribution (histogram, box plots)
- Work_Experience distribution
- Family_Size distribution
- Statistical summaries (mean, median, std, quartiles)
- Outlier detection

**5. Categorical Feature Analysis**
- Gender distribution
- Ever_Married distribution
- Graduated distribution
- Profession distribution (9 categories)
- Spending_Score distribution
- Var_1 distribution

**6. Feature Correlations**
- Correlation matrix for numerical features
- Heatmap visualization
- Relationship identification

**7. Segmentation Analysis**
- Feature distributions by segment
- Age distribution across segments
- Average feature values per segment
- Spending score by segment

**8. Key Insights Summary**
- Dataset statistics
- Missing value summary
- Feature importance indicators
- Recommended next steps

---

*[Continued in next file due to length...]*

Would you like me to continue with the remaining sections (8-12)?
