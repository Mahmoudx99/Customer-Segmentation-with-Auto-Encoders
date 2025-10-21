"""
Script to download the customer segmentation dataset from Kaggle.

Before running:
1. Install kaggle: pip install kaggle
2. Place your kaggle.json in ~/.kaggle/
3. Run: chmod 600 ~/.kaggle/kaggle.json
"""
import os
import subprocess


def download_dataset():
    """Download dataset from Kaggle."""
    # Dataset URL: https://www.kaggle.com/datasets/vetrirah/customer/
    dataset_name = "vetrirah/customer"
    output_dir = "data/raw"

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    print(f"Downloading dataset: {dataset_name}")
    print(f"Output directory: {output_dir}")

    try:
        # Download using Kaggle API
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", dataset_name, "-p", output_dir, "--unzip"],
            check=True
        )
        print("\nDataset downloaded successfully!")

        # List downloaded files
        files = os.listdir(output_dir)
        print(f"\nFiles in {output_dir}:")
        for file in files:
            file_path = os.path.join(output_dir, file)
            size = os.path.getsize(file_path) / 1024  # Size in KB
            print(f"  - {file} ({size:.2f} KB)")

    except subprocess.CalledProcessError as e:
        print(f"\nError downloading dataset: {e}")
        print("\nMake sure you have:")
        print("1. Installed kaggle: pip install kaggle")
        print("2. Placed kaggle.json in ~/.kaggle/")
        print("3. Set permissions: chmod 600 ~/.kaggle/kaggle.json")
        print("4. Accepted dataset terms on Kaggle website")
    except FileNotFoundError:
        print("\nKaggle CLI not found. Please install it:")
        print("pip install kaggle")


if __name__ == "__main__":
    download_dataset()
