"""
Script to download heart disease dataset from UCI ML Repository.
"""
import argparse
import hashlib
import logging
import os
import urllib.request
from datetime import datetime

import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def compute_file_hash(file_path):
    """Compute SHA-256 hash of file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def download_dataset(output_dir="data/raw", validate_hash=True):
    """
    Download heart disease dataset from UCI ML Repository.

    Args:
        output_dir: Directory to save the raw data
        validate_hash: Whether to validate file hash

    Returns:
        Path to the downloaded dataset
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # URLs for the dataset
    base_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease"
    urls = {
        "cleveland": f"{base_url}/processed.cleveland.data",
        "hungarian": f"{base_url}/processed.hungarian.data",
        "switzerland": f"{base_url}/processed.switzerland.data",
        "va": f"{base_url}/processed.va.data",
    }

    # Expected SHA-256 hashes for validation
    expected_hashes = {
        "cleveland": "3e2093b6a81fe5e6a2169ace8d19c9a9939251fe9ac3a24b779bcf4d2c0a72ee",
        "hungarian": "f982fa083fd587f42f0e59c767ae59a8f37b16d22c87db1dec7e8a8df5b66900",
        "switzerland": (
            "5d6a616f9e32a478c6b93e5d162a9ce6a3b91e3c736c6c6c9ea5c838659c42a9"
        ),
        "va": "de0c3320a94f5d4de567c97d2f8fd203a478b1f1da95e9c3ea4788c25c884bd9",
    }

    # Column names for the dataset
    column_names = [
        "age",
        "sex",
        "cp",
        "trestbps",
        "chol",
        "fbs",
        "restecg",
        "thalach",
        "exang",
        "oldpeak",
        "slope",
        "ca",
        "thal",
        "target",
    ]

    downloaded_files = {}

    # Download each dataset
    for dataset_name, url in urls.items():
        output_file = os.path.join(output_dir, f"{dataset_name}.csv")
        logger.info(f"Downloading {dataset_name} dataset to {output_file}")

        try:
            urllib.request.urlretrieve(url, output_file)
            logger.info(f"Successfully downloaded {dataset_name} dataset")

            # Validate file hash
            if validate_hash:
                file_hash = compute_file_hash(output_file)
                if file_hash == expected_hashes[dataset_name]:
                    logger.info(
                        f"Hash validation successful for {dataset_name} dataset"
                    )
                else:
                    logger.warning(
                        f"Hash validation failed for {dataset_name} dataset. "
                        f"Expected: {expected_hashes[dataset_name]}, "
                        f"Got: {file_hash}"
                    )

            # Read the data and add column names
            df = pd.read_csv(
                output_file, header=None, names=column_names, na_values="?"
            )

            # Save with column names
            df.to_csv(output_file, index=False)

            downloaded_files[dataset_name] = output_file

        except Exception as e:
            logger.error(f"Error downloading {dataset_name} dataset: {e}")

    # Create a combined dataset
    if len(downloaded_files) > 0:
        combined_file = os.path.join(output_dir, "heart_disease_combined.csv")
        logger.info(f"Creating combined dataset at {combined_file}")

        dfs = []
        for dataset_name, file_path in downloaded_files.items():
            df = pd.read_csv(file_path)
            df["source"] = dataset_name
            dfs.append(df)

        combined_df = pd.concat(dfs, ignore_index=True)
        combined_df.to_csv(combined_file, index=False)
        logger.info(
            f"Combined dataset created with {len(combined_df)} records"
        )

        # Create metadata file
        metadata = {
            "download_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "dataset_sources": list(urls.keys()),
            "total_records": len(combined_df),
            "columns": list(combined_df.columns),
            "missing_values": combined_df.isnull().sum().to_dict(),
        }

        metadata_file = os.path.join(output_dir, "metadata.txt")
        with open(metadata_file, "w") as f:
            for key, value in metadata.items():
                f.write(f"{key}: {value}\n")

        return combined_file

    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download heart disease dataset"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/raw",
        help="Directory to save the raw data",
    )
    args = parser.parse_args()

    download_dataset(output_dir=args.output_dir)