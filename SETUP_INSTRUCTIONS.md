# Setup Instructions

## Step 1: Download the Dataset

### Option A: Using Kaggle API (Recommended)

1. **Get your Kaggle API credentials:**
   - Go to https://www.kaggle.com
   - Click on your profile picture → Settings
   - Scroll to "API" section → Click "Create New API Token"
   - This downloads `kaggle.json`

2. **Configure Kaggle API:**
   ```bash
   mkdir -p ~/.kaggle
   mv ~/Downloads/kaggle.json ~/.kaggle/
   chmod 600 ~/.kaggle/kaggle.json
   ```

3. **Download the dataset:**
   ```bash
   python download_data.py
   ```

### Option B: Manual Download

1. Visit: https://www.kaggle.com/datasets/vetrirah/customer/
2. Click "Download" button
3. Extract the ZIP file
4. Place `Train.csv` and `Test.csv` in the `data/raw/` directory

## Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

## Step 3: Verify Setup

Run the verification script:
```bash
python -c "from src.data_loader import load_data; load_data()"
```

You should see the shape of training and test datasets printed.

## Step 4: Run the Analysis

```bash
# Start Jupyter Notebook
jupyter notebook

# Open notebooks/01_eda.ipynb
```

## Next Steps

Once setup is complete, proceed with:
1. Exploratory Data Analysis (EDA)
2. Data Preprocessing
3. Auto Encoder Training
4. Customer Segmentation
5. Evaluation and Predictions
