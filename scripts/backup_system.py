#!/usr/bin/env python3
"""
Backup and recovery system for the Heart Disease Prediction project.

This script provides functionality to back up critical data to a local directory
and optionally to cloud storage (AWS S3, Azure Blob Storage, or Google Cloud Storage).

Usage:
    backup_system.py backup [--cloud] [--storage {s3,azure,gcp}]
    backup_system.py restore [--cloud] [--storage {s3,azure,gcp}] [--timestamp TIMESTAMP]
    backup_system.py list [--cloud] [--storage {s3,azure,gcp}]
    backup_system.py prune [--keep N] [--cloud] [--storage {s3,azure,gcp}]
"""

import argparse
import datetime
import glob
import hashlib
import json
import logging
import os
import shutil
import sys
import tarfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/backup.log"),
    ],
)
logger = logging.getLogger("backup_system")

# Make sure logs directory exists
os.makedirs("logs", exist_ok=True)

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
BACKUP_DIR = PROJECT_ROOT / "backups"
BACKUP_MANIFEST_FILE = BACKUP_DIR / "manifest.json"
CLOUD_CREDENTIALS_FILE = PROJECT_ROOT / "config" / "cloud_credentials.json"

# Default paths to back up (relative to project root)
DEFAULT_BACKUP_PATHS = [
    "models/keras_mlp_model.h5",
    "models/sklearn_mlp_model.joblib",
    "models/evaluation_results.joblib",
    "models/optuna/keras_mlp_study.pkl",
    "models/optuna/sklearn_mlp_study.pkl",
    "data/processed/preprocessor.joblib",
    "data/processed/original_splits.joblib",
    "data/processed/processed_data.npz",
    "data/processed/processing_metadata.txt",
    "config/config.yaml",
]


def get_file_hash(filepath: Path) -> str:
    """Compute MD5 hash of a file."""
    md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            md5.update(chunk)
    return md5.hexdigest()


def create_backup(cloud: bool = False, storage: str = None) -> str:
    """
    Create a backup of the system.

    Args:
        cloud: Whether to upload to cloud storage
        storage: Cloud storage provider (s3, azure, gcp)

    Returns:
        Backup timestamp
    """
    # Create backup timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = BACKUP_DIR / timestamp
    os.makedirs(backup_path, exist_ok=True)

    logger.info(f"Creating backup with timestamp: {timestamp}")
    logger.info(f"Backup directory: {backup_path}")

    # Store file metadata and hashes
    manifest = {
        "timestamp": timestamp,
        "creation_time": datetime.datetime.now().isoformat(),
        "files": {},
        "missing_files": [],
    }

    # Copy files to backup directory
    for rel_path in DEFAULT_BACKUP_PATHS:
        source_path = PROJECT_ROOT / rel_path
        dest_path = backup_path / rel_path

        # Create destination directory if it doesn't exist
        os.makedirs(dest_path.parent, exist_ok=True)

        if source_path.exists():
            # Copy file to backup directory
            shutil.copy2(source_path, dest_path)

            # Calculate and store file hash
            file_hash = get_file_hash(source_path)
            file_size = source_path.stat().st_size
            manifest["files"][rel_path] = {
                "hash": file_hash,
                "size": file_size,
                "backed_up": True,
            }
            logger.info(f"Backed up: {rel_path} ({file_size} bytes)")
        else:
            logger.warning(f"File not found: {rel_path}")
            manifest["missing_files"].append(rel_path)

    # Create a compressed archive
    archive_path = BACKUP_DIR / f"backup_{timestamp}.tar.gz"
    with tarfile.open(archive_path, "w:gz") as tar:
        tar.add(backup_path, arcname=timestamp)

    # Add archive info to manifest
    manifest["archive"] = {
        "path": str(archive_path.relative_to(PROJECT_ROOT)),
        "size": archive_path.stat().st_size,
        "hash": get_file_hash(archive_path),
    }

    # Store manifest in backup directory
    with open(backup_path / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    # Update global manifest
    update_global_manifest(manifest)

    logger.info(f"Backup archive created: {archive_path}")

    # Clean up temporary backup directory
    shutil.rmtree(backup_path)

    # Upload to cloud if requested
    if cloud:
        if not storage:
            logger.error("Cloud storage provider not specified")
            return timestamp

        upload_to_cloud(archive_path, storage)

    return timestamp


def update_global_manifest(manifest: Dict) -> None:
    """Update the global backup manifest."""
    # Create global manifest if it doesn't exist
    if not BACKUP_MANIFEST_FILE.exists():
        with open(BACKUP_MANIFEST_FILE, "w") as f:
            json.dump({"backups": []}, f, indent=2)

    # Load existing manifest
    with open(BACKUP_MANIFEST_FILE, "r") as f:
        global_manifest = json.load(f)

    # Add new backup to manifest
    global_manifest["backups"].append(
        {
            "timestamp": manifest["timestamp"],
            "creation_time": manifest["creation_time"],
            "archive": manifest["archive"]["path"],
            "archive_size": manifest["archive"]["size"],
            "archive_hash": manifest["archive"]["hash"],
            "file_count": len(manifest["files"]),
            "missing_files": len(manifest["missing_files"]),
        }
    )

    # Sort backups by timestamp (newest first)
    global_manifest["backups"].sort(key=lambda x: x["timestamp"], reverse=True)

    # Save updated manifest
    with open(BACKUP_MANIFEST_FILE, "w") as f:
        json.dump(global_manifest, f, indent=2)


def upload_to_cloud(archive_path: Path, storage: str) -> bool:
    """
    Upload backup archive to cloud storage.

    Args:
        archive_path: Path to backup archive
        storage: Cloud storage provider (s3, azure, gcp)

    Returns:
        Success status
    """
    if not CLOUD_CREDENTIALS_FILE.exists():
        logger.error(f"Cloud credentials file not found: {CLOUD_CREDENTIALS_FILE}")
        return False

    try:
        # Load cloud credentials
        with open(CLOUD_CREDENTIALS_FILE, "r") as f:
            credentials = json.load(f)

        if storage == "s3":
            return upload_to_s3(archive_path, credentials)
        elif storage == "azure":
            return upload_to_azure(archive_path, credentials)
        elif storage == "gcp":
            return upload_to_gcp(archive_path, credentials)
        else:
            logger.error(f"Unsupported cloud storage provider: {storage}")
            return False

    except Exception as e:
        logger.error(f"Error uploading to cloud: {e}")
        return False


def upload_to_s3(archive_path: Path, credentials: Dict) -> bool:
    """Upload backup archive to AWS S3."""
    try:
        import boto3
        from botocore.exceptions import ClientError

        logger.info("Uploading to AWS S3...")

        # Extract S3 credentials
        aws_access_key = credentials.get("aws_access_key_id")
        aws_secret_key = credentials.get("aws_secret_access_key")
        bucket_name = credentials.get("s3_bucket")

        if not (aws_access_key and aws_secret_key and bucket_name):
            logger.error("Missing required S3 credentials")
            return False

        # Create S3 client
        s3_client = boto3.client(
            "s3", aws_access_key_id=aws_access_key, aws_secret_access_key=aws_secret_key
        )

        # Upload file
        object_name = f"heart_disease_mlp/backups/{archive_path.name}"
        s3_client.upload_file(str(archive_path), bucket_name, object_name)

        logger.info(f"Successfully uploaded to S3: {bucket_name}/{object_name}")
        return True

    except ImportError:
        logger.error("boto3 not installed. Run 'pip install boto3' to enable S3 uploads.")
        return False
    except ClientError as e:
        logger.error(f"S3 upload error: {e}")
        return False
    except Exception as e:
        logger.error(f"S3 upload error: {e}")
        return False


def upload_to_azure(archive_path: Path, credentials: Dict) -> bool:
    """Upload backup archive to Azure Blob Storage."""
    try:
        from azure.storage.blob import BlobServiceClient

        logger.info("Uploading to Azure Blob Storage...")

        # Extract Azure credentials
        connection_string = credentials.get("azure_connection_string")
        container_name = credentials.get("azure_container")

        if not (connection_string and container_name):
            logger.error("Missing required Azure credentials")
            return False

        # Create Azure client
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        container_client = blob_service_client.get_container_client(container_name)

        # Upload file
        blob_name = f"heart_disease_mlp/backups/{archive_path.name}"
        with open(archive_path, "rb") as data:
            container_client.upload_blob(name=blob_name, data=data, overwrite=True)

        logger.info(f"Successfully uploaded to Azure: {container_name}/{blob_name}")
        return True

    except ImportError:
        logger.error(
            "azure-storage-blob not installed. Run 'pip install azure-storage-blob' to enable Azure uploads."
        )
        return False
    except Exception as e:
        logger.error(f"Azure upload error: {e}")
        return False


def upload_to_gcp(archive_path: Path, credentials: Dict) -> bool:
    """Upload backup archive to Google Cloud Storage."""
    try:
        from google.cloud import storage

        logger.info("Uploading to Google Cloud Storage...")

        # Extract GCP credentials
        bucket_name = credentials.get("gcp_bucket")

        if not bucket_name:
            logger.error("Missing required GCP credentials")
            return False

        # Create GCP client
        # Note: Uses ADC (Application Default Credentials)
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)

        # Upload file
        blob_name = f"heart_disease_mlp/backups/{archive_path.name}"
        blob = bucket.blob(blob_name)
        blob.upload_from_filename(str(archive_path))

        logger.info(f"Successfully uploaded to GCP: {bucket_name}/{blob_name}")
        return True

    except ImportError:
        logger.error(
            "google-cloud-storage not installed. Run 'pip install google-cloud-storage' to enable GCP uploads."
        )
        return False
    except Exception as e:
        logger.error(f"GCP upload error: {e}")
        return False


def download_from_cloud(timestamp: str, storage: str) -> Optional[Path]:
    """
    Download backup archive from cloud storage.

    Args:
        timestamp: Backup timestamp
        storage: Cloud storage provider (s3, azure, gcp)

    Returns:
        Path to downloaded archive or None if download failed
    """
    try:
        # Load cloud credentials
        with open(CLOUD_CREDENTIALS_FILE, "r") as f:
            credentials = json.load(f)

        archive_name = f"backup_{timestamp}.tar.gz"
        download_path = BACKUP_DIR / archive_name

        if storage == "s3":
            return download_from_s3(timestamp, download_path, credentials)
        elif storage == "azure":
            return download_from_azure(timestamp, download_path, credentials)
        elif storage == "gcp":
            return download_from_gcp(timestamp, download_path, credentials)
        else:
            logger.error(f"Unsupported cloud storage provider: {storage}")
            return None

    except Exception as e:
        logger.error(f"Error downloading from cloud: {e}")
        return None


def download_from_s3(timestamp: str, download_path: Path, credentials: Dict) -> Optional[Path]:
    """Download backup archive from AWS S3."""
    try:
        import boto3
        from botocore.exceptions import ClientError

        logger.info(f"Downloading from AWS S3: {timestamp}")

        # Extract S3 credentials
        aws_access_key = credentials.get("aws_access_key_id")
        aws_secret_key = credentials.get("aws_secret_access_key")
        bucket_name = credentials.get("s3_bucket")

        if not (aws_access_key and aws_secret_key and bucket_name):
            logger.error("Missing required S3 credentials")
            return None

        # Create S3 client
        s3_client = boto3.client(
            "s3", aws_access_key_id=aws_access_key, aws_secret_access_key=aws_secret_key
        )

        # Download file
        object_name = f"heart_disease_mlp/backups/{download_path.name}"
        s3_client.download_file(bucket_name, object_name, str(download_path))

        logger.info(f"Successfully downloaded from S3: {download_path}")
        return download_path

    except ImportError:
        logger.error("boto3 not installed. Run 'pip install boto3' to enable S3 downloads.")
        return None
    except ClientError as e:
        logger.error(f"S3 download error: {e}")
        return None
    except Exception as e:
        logger.error(f"S3 download error: {e}")
        return None


def download_from_azure(timestamp: str, download_path: Path, credentials: Dict) -> Optional[Path]:
    """Download backup archive from Azure Blob Storage."""
    try:
        from azure.storage.blob import BlobServiceClient

        logger.info(f"Downloading from Azure Blob Storage: {timestamp}")

        # Extract Azure credentials
        connection_string = credentials.get("azure_connection_string")
        container_name = credentials.get("azure_container")

        if not (connection_string and container_name):
            logger.error("Missing required Azure credentials")
            return None

        # Create Azure client
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        container_client = blob_service_client.get_container_client(container_name)

        # Download file
        blob_name = f"heart_disease_mlp/backups/{download_path.name}"
        blob_client = container_client.get_blob_client(blob_name)

        with open(download_path, "wb") as f:
            f.write(blob_client.download_blob().readall())

        logger.info(f"Successfully downloaded from Azure: {download_path}")
        return download_path

    except ImportError:
        logger.error(
            "azure-storage-blob not installed. Run 'pip install azure-storage-blob' to enable Azure downloads."
        )
        return None
    except Exception as e:
        logger.error(f"Azure download error: {e}")
        return None


def download_from_gcp(timestamp: str, download_path: Path, credentials: Dict) -> Optional[Path]:
    """Download backup archive from Google Cloud Storage."""
    try:
        from google.cloud import storage

        logger.info(f"Downloading from Google Cloud Storage: {timestamp}")

        # Extract GCP credentials
        bucket_name = credentials.get("gcp_bucket")

        if not bucket_name:
            logger.error("Missing required GCP credentials")
            return None

        # Create GCP client
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)

        # Download file
        blob_name = f"heart_disease_mlp/backups/{download_path.name}"
        blob = bucket.blob(blob_name)
        blob.download_to_filename(str(download_path))

        logger.info(f"Successfully downloaded from GCP: {download_path}")
        return download_path

    except ImportError:
        logger.error(
            "google-cloud-storage not installed. Run 'pip install google-cloud-storage' to enable GCP downloads."
        )
        return None
    except Exception as e:
        logger.error(f"GCP download error: {e}")
        return None


def list_backups(cloud: bool = False, storage: str = None) -> List[Dict]:
    """
    List available backups.

    Args:
        cloud: Whether to list cloud backups
        storage: Cloud storage provider (s3, azure, gcp)

    Returns:
        List of backup metadata
    """
    if not BACKUP_MANIFEST_FILE.exists():
        logger.info("No backups found.")
        return []

    with open(BACKUP_MANIFEST_FILE, "r") as f:
        manifest = json.load(f)

    backups = manifest.get("backups", [])

    # Check if backups exist
    if len(backups) == 0:
        logger.info("No backups found.")
        return []

    # Print backup information
    logger.info(f"Found {len(backups)} backup(s):")
    for idx, backup in enumerate(backups):
        timestamp = backup["timestamp"]
        creation_time = backup["creation_time"]
        file_count = backup["file_count"]
        archive_size = backup["archive_size"]

        # Convert size to human-readable format
        size_str = f"{archive_size / (1024*1024):.2f} MB"

        logger.info(
            f"{idx+1}. Timestamp: {timestamp} | Created: {creation_time} | Files: {file_count} | Size: {size_str}"
        )

    # If cloud is specified, also check cloud storage
    if cloud and storage:
        list_cloud_backups(storage)

    return backups


def list_cloud_backups(storage: str) -> None:
    """List backups in cloud storage."""
    try:
        # Load cloud credentials
        with open(CLOUD_CREDENTIALS_FILE, "r") as f:
            credentials = json.load(f)

        if storage == "s3":
            list_s3_backups(credentials)
        elif storage == "azure":
            list_azure_backups(credentials)
        elif storage == "gcp":
            list_gcp_backups(credentials)
        else:
            logger.error(f"Unsupported cloud storage provider: {storage}")

    except FileNotFoundError:
        logger.error(f"Cloud credentials file not found: {CLOUD_CREDENTIALS_FILE}")
    except Exception as e:
        logger.error(f"Error listing cloud backups: {e}")


def list_s3_backups(credentials: Dict) -> None:
    """List backups in AWS S3."""
    try:
        import boto3

        # Extract S3 credentials
        aws_access_key = credentials.get("aws_access_key_id")
        aws_secret_key = credentials.get("aws_secret_access_key")
        bucket_name = credentials.get("s3_bucket")

        if not (aws_access_key and aws_secret_key and bucket_name):
            logger.error("Missing required S3 credentials")
            return

        # Create S3 client
        s3_client = boto3.client(
            "s3", aws_access_key_id=aws_access_key, aws_secret_access_key=aws_secret_key
        )

        # List objects in bucket
        response = s3_client.list_objects_v2(
            Bucket=bucket_name, Prefix="heart_disease_mlp/backups/"
        )

        if "Contents" not in response:
            logger.info("No backups found in S3.")
            return

        # Print backup information
        logger.info(f"Found {len(response['Contents'])} backup(s) in S3:")
        for idx, obj in enumerate(response["Contents"]):
            key = obj["Key"]
            size = obj["Size"]
            last_modified = obj["LastModified"]

            # Extract timestamp from key
            # Format: heart_disease_mlp/backups/backup_YYYYMMDD_HHMMSS.tar.gz
            filename = key.split("/")[-1]
            timestamp = filename.replace("backup_", "").replace(".tar.gz", "")

            # Convert size to human-readable format
            size_str = f"{size / (1024*1024):.2f} MB"

            logger.info(
                f"{idx+1}. Timestamp: {timestamp} | Modified: {last_modified} | Size: {size_str}"
            )

    except ImportError:
        logger.error("boto3 not installed. Run 'pip install boto3' to enable S3 operations.")
    except Exception as e:
        logger.error(f"S3 error: {e}")


def list_azure_backups(credentials: Dict) -> None:
    """List backups in Azure Blob Storage."""
    try:
        from azure.storage.blob import BlobServiceClient

        # Extract Azure credentials
        connection_string = credentials.get("azure_connection_string")
        container_name = credentials.get("azure_container")

        if not (connection_string and container_name):
            logger.error("Missing required Azure credentials")
            return

        # Create Azure client
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        container_client = blob_service_client.get_container_client(container_name)

        # List blobs in container
        blobs = container_client.list_blobs(name_starts_with="heart_disease_mlp/backups/")
        blobs_list = list(blobs)

        if not blobs_list:
            logger.info("No backups found in Azure.")
            return

        # Print backup information
        logger.info(f"Found {len(blobs_list)} backup(s) in Azure:")
        for idx, blob in enumerate(blobs_list):
            name = blob.name
            size = blob.size
            last_modified = blob.last_modified

            # Extract timestamp from name
            # Format: heart_disease_mlp/backups/backup_YYYYMMDD_HHMMSS.tar.gz
            filename = name.split("/")[-1]
            timestamp = filename.replace("backup_", "").replace(".tar.gz", "")

            # Convert size to human-readable format
            size_str = f"{size / (1024*1024):.2f} MB"

            logger.info(
                f"{idx+1}. Timestamp: {timestamp} | Modified: {last_modified} | Size: {size_str}"
            )

    except ImportError:
        logger.error(
            "azure-storage-blob not installed. Run 'pip install azure-storage-blob' to enable Azure operations."
        )
    except Exception as e:
        logger.error(f"Azure error: {e}")


def list_gcp_backups(credentials: Dict) -> None:
    """List backups in Google Cloud Storage."""
    try:
        from google.cloud import storage

        # Extract GCP credentials
        bucket_name = credentials.get("gcp_bucket")

        if not bucket_name:
            logger.error("Missing required GCP credentials")
            return

        # Create GCP client
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)

        # List blobs in bucket
        blobs = list(bucket.list_blobs(prefix="heart_disease_mlp/backups/"))

        if not blobs:
            logger.info("No backups found in GCP.")
            return

        # Print backup information
        logger.info(f"Found {len(blobs)} backup(s) in GCP:")
        for idx, blob in enumerate(blobs):
            name = blob.name
            size = blob.size
            last_modified = blob.updated

            # Extract timestamp from name
            # Format: heart_disease_mlp/backups/backup_YYYYMMDD_HHMMSS.tar.gz
            filename = name.split("/")[-1]
            timestamp = filename.replace("backup_", "").replace(".tar.gz", "")

            # Convert size to human-readable format
            size_str = f"{size / (1024*1024):.2f} MB"

            logger.info(
                f"{idx+1}. Timestamp: {timestamp} | Modified: {last_modified} | Size: {size_str}"
            )

    except ImportError:
        logger.error(
            "google-cloud-storage not installed. Run 'pip install google-cloud-storage' to enable GCP operations."
        )
    except Exception as e:
        logger.error(f"GCP error: {e}")


def restore_backup(timestamp: str = None, cloud: bool = False, storage: str = None) -> bool:
    """
    Restore system from a backup.

    Args:
        timestamp: Backup timestamp to restore (if None, use latest)
        cloud: Whether to download from cloud storage
        storage: Cloud storage provider (s3, azure, gcp)

    Returns:
        Success status
    """
    # If no timestamp specified, use latest
    if timestamp is None:
        backups = list_backups(cloud=False)
        if not backups:
            logger.error("No backups found.")
            return False

        timestamp = backups[0]["timestamp"]
        logger.info(f"Using latest backup: {timestamp}")

    archive_path = BACKUP_DIR / f"backup_{timestamp}.tar.gz"

    # If archive doesn't exist locally and cloud is specified, download from cloud
    if not archive_path.exists() and cloud:
        if not storage:
            logger.error("Cloud storage provider not specified")
            return False

        archive_path = download_from_cloud(timestamp, storage)
        if not archive_path:
            logger.error(f"Failed to download backup {timestamp} from cloud")
            return False

    # Check if archive exists
    if not archive_path.exists():
        logger.error(f"Backup archive not found: {archive_path}")
        return False

    logger.info(f"Restoring backup: {timestamp}")

    # Extract archive to temporary directory
    temp_dir = BACKUP_DIR / f"temp_{timestamp}"
    os.makedirs(temp_dir, exist_ok=True)

    try:
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(path=temp_dir)

        # Load manifest
        manifest_path = temp_dir / timestamp / "manifest.json"
        with open(manifest_path, "r") as f:
            manifest = json.load(f)

        # Restore files
        for rel_path, file_info in manifest["files"].items():
            source_path = temp_dir / timestamp / rel_path
            dest_path = PROJECT_ROOT / rel_path

            # Create destination directory if it doesn't exist
            os.makedirs(dest_path.parent, exist_ok=True)

            # Copy file to destination
            if source_path.exists():
                logger.info(f"Restoring: {rel_path}")
                shutil.copy2(source_path, dest_path)
            else:
                logger.warning(f"File not found in backup: {rel_path}")

        logger.info(f"Backup {timestamp} restored successfully")
        return True

    except Exception as e:
        logger.error(f"Error restoring backup: {e}")
        return False
    finally:
        # Clean up temporary directory
        shutil.rmtree(temp_dir)


def prune_backups(keep: int = 5, cloud: bool = False, storage: str = None) -> bool:
    """
    Prune old backups, keeping only the specified number of most recent backups.

    Args:
        keep: Number of most recent backups to keep
        cloud: Whether to prune cloud backups
        storage: Cloud storage provider (s3, azure, gcp)

    Returns:
        Success status
    """
    if not BACKUP_MANIFEST_FILE.exists():
        logger.info("No backups found.")
        return True

    with open(BACKUP_MANIFEST_FILE, "r") as f:
        manifest = json.load(f)

    backups = manifest.get("backups", [])

    # Check if backups exist
    if len(backups) == 0:
        logger.info("No backups found.")
        return True

    # If there are more backups than we want to keep
    if len(backups) > keep:
        # Keep the most recent 'keep' backups
        backups_to_keep = backups[:keep]
        backups_to_delete = backups[keep:]

        logger.info(
            f"Keeping {len(backups_to_keep)} most recent backups, deleting {len(backups_to_delete)} older backups."
        )

        # Delete local archives
        for backup in backups_to_delete:
            archive_path = PROJECT_ROOT / backup["archive"]

            if archive_path.exists():
                logger.info(f"Deleting backup: {backup['timestamp']}")
                os.remove(archive_path)

        # Update global manifest
        manifest["backups"] = backups_to_keep
        with open(BACKUP_MANIFEST_FILE, "w") as f:
            json.dump(manifest, f, indent=2)

        # Delete cloud backups if requested
        if cloud and storage:
            for backup in backups_to_delete:
                delete_cloud_backup(backup["timestamp"], storage)

    else:
        logger.info(
            f"Only {len(backups)} backups exist, which is less than or equal to the {keep} backups to keep. No pruning needed."
        )

    return True


def delete_cloud_backup(timestamp: str, storage: str) -> bool:
    """
    Delete backup from cloud storage.

    Args:
        timestamp: Backup timestamp
        storage: Cloud storage provider (s3, azure, gcp)

    Returns:
        Success status
    """
    try:
        # Load cloud credentials
        with open(CLOUD_CREDENTIALS_FILE, "r") as f:
            credentials = json.load(f)

        if storage == "s3":
            return delete_s3_backup(timestamp, credentials)
        elif storage == "azure":
            return delete_azure_backup(timestamp, credentials)
        elif storage == "gcp":
            return delete_gcp_backup(timestamp, credentials)
        else:
            logger.error(f"Unsupported cloud storage provider: {storage}")
            return False

    except Exception as e:
        logger.error(f"Error deleting cloud backup: {e}")
        return False


def delete_s3_backup(timestamp: str, credentials: Dict) -> bool:
    """Delete backup from AWS S3."""
    try:
        import boto3

        logger.info(f"Deleting backup from AWS S3: {timestamp}")

        # Extract S3 credentials
        aws_access_key = credentials.get("aws_access_key_id")
        aws_secret_key = credentials.get("aws_secret_access_key")
        bucket_name = credentials.get("s3_bucket")

        if not (aws_access_key and aws_secret_key and bucket_name):
            logger.error("Missing required S3 credentials")
            return False

        # Create S3 client
        s3_client = boto3.client(
            "s3", aws_access_key_id=aws_access_key, aws_secret_access_key=aws_secret_key
        )

        # Delete file
        object_name = f"heart_disease_mlp/backups/backup_{timestamp}.tar.gz"
        s3_client.delete_object(Bucket=bucket_name, Key=object_name)

        logger.info(f"Successfully deleted from S3: {bucket_name}/{object_name}")
        return True

    except ImportError:
        logger.error("boto3 not installed. Run 'pip install boto3' to enable S3 operations.")
        return False
    except Exception as e:
        logger.error(f"S3 delete error: {e}")
        return False


def delete_azure_backup(timestamp: str, credentials: Dict) -> bool:
    """Delete backup from Azure Blob Storage."""
    try:
        from azure.storage.blob import BlobServiceClient

        logger.info(f"Deleting backup from Azure Blob Storage: {timestamp}")

        # Extract Azure credentials
        connection_string = credentials.get("azure_connection_string")
        container_name = credentials.get("azure_container")

        if not (connection_string and container_name):
            logger.error("Missing required Azure credentials")
            return False

        # Create Azure client
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        container_client = blob_service_client.get_container_client(container_name)

        # Delete file
        blob_name = f"heart_disease_mlp/backups/backup_{timestamp}.tar.gz"
        container_client.delete_blob(blob_name)

        logger.info(f"Successfully deleted from Azure: {container_name}/{blob_name}")
        return True

    except ImportError:
        logger.error(
            "azure-storage-blob not installed. Run 'pip install azure-storage-blob' to enable Azure operations."
        )
        return False
    except Exception as e:
        logger.error(f"Azure delete error: {e}")
        return False


def delete_gcp_backup(timestamp: str, credentials: Dict) -> bool:
    """Delete backup from Google Cloud Storage."""
    try:
        from google.cloud import storage

        logger.info(f"Deleting backup from Google Cloud Storage: {timestamp}")

        # Extract GCP credentials
        bucket_name = credentials.get("gcp_bucket")

        if not bucket_name:
            logger.error("Missing required GCP credentials")
            return False

        # Create GCP client
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)

        # Delete file
        blob_name = f"heart_disease_mlp/backups/backup_{timestamp}.tar.gz"
        blob = bucket.blob(blob_name)
        blob.delete()

        logger.info(f"Successfully deleted from GCP: {bucket_name}/{blob_name}")
        return True

    except ImportError:
        logger.error(
            "google-cloud-storage not installed. Run 'pip install google-cloud-storage' to enable GCP operations."
        )
        return False
    except Exception as e:
        logger.error(f"GCP delete error: {e}")
        return False


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Backup and recovery system for Heart Disease Prediction project"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Backup command
    backup_parser = subparsers.add_parser("backup", help="Create a new backup")
    backup_parser.add_argument(
        "--cloud", action="store_true", help="Upload backup to cloud storage"
    )
    backup_parser.add_argument(
        "--storage", choices=["s3", "azure", "gcp"], help="Cloud storage provider"
    )

    # Restore command
    restore_parser = subparsers.add_parser("restore", help="Restore from a backup")
    restore_parser.add_argument("--timestamp", help="Backup timestamp to restore (default: latest)")
    restore_parser.add_argument(
        "--cloud", action="store_true", help="Download backup from cloud storage"
    )
    restore_parser.add_argument(
        "--storage", choices=["s3", "azure", "gcp"], help="Cloud storage provider"
    )

    # List command
    list_parser = subparsers.add_parser("list", help="List available backups")
    list_parser.add_argument("--cloud", action="store_true", help="List cloud backups")
    list_parser.add_argument(
        "--storage", choices=["s3", "azure", "gcp"], help="Cloud storage provider"
    )

    # Prune command
    prune_parser = subparsers.add_parser("prune", help="Prune old backups")
    prune_parser.add_argument(
        "--keep", type=int, default=5, help="Number of most recent backups to keep"
    )
    prune_parser.add_argument("--cloud", action="store_true", help="Prune cloud backups")
    prune_parser.add_argument(
        "--storage", choices=["s3", "azure", "gcp"], help="Cloud storage provider"
    )

    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()

    # Create backup directory if it doesn't exist
    os.makedirs(BACKUP_DIR, exist_ok=True)

    if args.command == "backup":
        create_backup(args.cloud, args.storage)
    elif args.command == "restore":
        restore_backup(args.timestamp, args.cloud, args.storage)
    elif args.command == "list":
        list_backups(args.cloud, args.storage)
    elif args.command == "prune":
        prune_backups(args.keep, args.cloud, args.storage)
    else:
        logger.error("No command specified. Use --help for usage information.")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
