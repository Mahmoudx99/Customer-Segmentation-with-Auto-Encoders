# Customer Segmentation with Auto Encoders

## Project Overview
This project implements customer segmentation using Auto Encoders and clustering algorithms to classify customers into 7 distinct segments for an automobile company's market expansion strategy.

## Dataset
**Source:** [Kaggle - Customer Segmentation Dataset](https://www.kaggle.com/datasets/vetrirah/customer/)

### Features
- **ID:** Unique customer identifier
- **Gender:** Customer gender
- **Ever_Married:** Marital status
- **Age:** Customer age
- **Graduated:** Education status
- **Profession:** Customer profession
- **Work_Experience:** Years of work experience
- **Spending_Score:** Customer spending score
- **Family_Size:** Number of family members
- **Var_1:** Label variable (target - 7 categories: cat_1 to cat_7)

## Approach
1. **Data Preprocessing:** Handle missing values, encode categorical features, scale numerical features
2. **Auto Encoder:** Deep learning model to learn compressed customer representations
3. **Clustering:** K-Means clustering (k=7) on encoded features
4. **Evaluation:** Silhouette score, t-SNE/PCA visualization

## Project Structure
```
.
├── data/
│   ├── raw/              # Original datasets
│   └── processed/        # Preprocessed datasets
├── notebooks/            # Jupyter notebooks for EDA and experiments
├── src/                  # Source code
│   ├── autoencoder.py    # Auto Encoder model
│   ├── clustering.py     # Clustering utilities
│   ├── preprocessing.py  # Data preprocessing
│   └── data_loader.py    # Data loading utilities
├── models/               # Trained models
├── results/              # Predictions and visualizations
├── Dockerfile            # Docker image definition
├── docker-compose.yml    # Docker Compose configuration
├── docker-run.sh         # Docker helper script (Linux/Mac)
├── docker-run.bat        # Docker helper script (Windows)
├── Makefile              # Make commands for Docker
├── requirements.txt      # Python dependencies
└── README.md            # Project documentation
```

## Quick Start with Docker (Recommended)

The easiest way to run this project is using Docker, which provides an isolated environment with all dependencies pre-installed.

### Prerequisites
- [Docker](https://www.docker.com/products/docker-desktop) installed on your system

### Run Training Pipeline

**Linux/Mac:**
```bash
./docker-run.sh build    # Build Docker image
./docker-run.sh train    # Run training
```

**Windows:**
```cmd
docker-run.bat build     # Build Docker image
docker-run.bat train     # Run training
```

**Or using Make:**
```bash
make build
make train
```

**Or using Docker Compose:**
```bash
docker-compose up train
```

### View Results
```bash
ls results/              # Check output files
ls results/plots/        # Check visualizations
```

For detailed Docker instructions, see [DOCKER.md](DOCKER.md)

## Setup

> **Note:** Using Docker (see above) is the recommended approach. The manual setup below is for advanced users who want to install dependencies directly on their system.

### Manual Setup (Alternative to Docker)

#### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

#### 2. Get the Data
The dataset is already included in the `Kaggle/` folder. Copy it to the data directory:
```bash
cp Kaggle/*.csv data/raw/
```

Or download from Kaggle (see SETUP_INSTRUCTIONS.md for details).

## Usage

### Quick Start - Run Complete Pipeline
Train the Auto Encoder and generate predictions in one command:
```bash
python train_pipeline.py
```

This will:
- Load and preprocess the data
- Train an Auto Encoder to learn compressed representations
- Perform K-Means clustering (k=7) on encoded features
- Generate visualizations and evaluation metrics
- Create predictions for test data
- Save all models and results

### Step-by-Step Usage

#### 1. Exploratory Data Analysis
```bash
jupyter notebook notebooks/01_exploratory_data_analysis.ipynb
```

#### 2. Train the Model
```bash
python train_pipeline.py
```

Output:
- `models/autoencoder.h5` - Trained Auto Encoder
- `models/preprocessor.pkl` - Fitted data preprocessor
- `models/clusterer.pkl` - Trained K-Means clusterer
- `results/submission.csv` - Final predictions
- `results/plots/` - Visualizations

#### 3. Make Predictions on New Data
```bash
python predict.py --input data/raw/Test.csv --output results/predictions.csv
```

## Model Architecture

### Auto Encoder
```
Input (9 features) → Dense(16) → Dense(12) → Bottleneck(8) → Dense(12) → Dense(16) → Output (9 features)
```

The bottleneck layer learns an 8-dimensional compressed representation of customer features.

### Clustering
K-Means clustering with k=7 is applied to the 8-dimensional encoded features to segment customers.

## Results

After running the pipeline, you'll find:

### Models
- `models/autoencoder.h5` - Trained Auto Encoder
- `models/preprocessor.pkl` - Data preprocessor
- `models/clusterer.pkl` - K-Means clusterer

### Predictions
- `results/submission.csv` - Test predictions (ID, Segmentation)
- `results/test_predictions_detailed.csv` - Detailed test results
- `results/training_results.csv` - Training set predictions

### Visualizations
- `results/plots/training_history.png` - Auto Encoder training curves
- `results/plots/optimal_clusters.png` - Elbow method and silhouette analysis
- `results/plots/clusters_tsne.png` - t-SNE visualization of clusters
- `results/plots/clusters_pca.png` - PCA visualization of clusters
- `results/plots/cluster_distribution.png` - Cluster size distribution

### Analysis
- `results/cluster_characteristics.csv` - Average feature values per cluster
- `results/clustering_metrics.csv` - Clustering quality metrics

## Evaluation Metrics

The clustering quality is evaluated using:
- **Silhouette Score**: Measures cluster cohesion and separation (-1 to 1, higher is better)
- **Davies-Bouldin Score**: Average similarity between clusters (lower is better)
- **Calinski-Harabasz Score**: Ratio of between-cluster to within-cluster variance (higher is better)

## Customization

Edit `train_pipeline.py` to customize:
- `ENCODING_DIM`: Size of bottleneck layer (default: 8)
- `HIDDEN_LAYERS`: Auto Encoder architecture (default: [16, 12])
- `N_CLUSTERS`: Number of customer segments (default: 7)
- `EPOCHS`: Training epochs (default: 100)
- `BATCH_SIZE`: Batch size (default: 32)
