# Backup and Recovery Procedures

This document outlines the backup and recovery procedures for the Heart Disease Prediction system. It provides information on how to create backups, restore from backups, manage backup storage, and integrate with cloud storage services.

## Table of Contents

- [Overview](#overview)
- [Critical Data](#critical-data)
- [Backup System](#backup-system)
  - [Creating Backups](#creating-backups)
  - [Listing Backups](#listing-backups)
  - [Restoring Backups](#restoring-backups)
  - [Pruning Old Backups](#pruning-old-backups)
- [Cloud Storage Integration](#cloud-storage-integration)
  - [AWS S3](#aws-s3)
  - [Azure Blob Storage](#azure-blob-storage)
  - [Google Cloud Storage](#google-cloud-storage)
- [Automated Backups](#automated-backups)
- [Disaster Recovery](#disaster-recovery)
- [Best Practices](#best-practices)

## Overview

The Heart Disease Prediction system's backup and recovery functionality ensures that critical data and model artifacts can be reliably backed up and restored in case of data loss, corruption, or system failure. The system provides:

- Automatic identification of critical files to back up
- Versioned backups with timestamps
- File integrity verification with checksums
- Compression to minimize storage usage
- Cloud storage integration (AWS S3, Azure Blob Storage, Google Cloud Storage)
- Backup pruning to manage storage space

## Critical Data

The following components are considered critical and are automatically included in backups:

1. **Trained Models**
   - `models/keras_mlp_model.h5` - Keras neural network model
   - `models/sklearn_mlp_model.joblib` - scikit-learn MLP model
   - `models/evaluation_results.joblib` - Model evaluation metrics and results

2. **Hyperparameter Optimization**
   - `models/optuna/keras_mlp_study.pkl` - Optuna study results for Keras model
   - `models/optuna/sklearn_mlp_study.pkl` - Optuna study results for scikit-learn model

3. **Preprocessed Data**
   - `data/processed/preprocessor.joblib` - Preprocessing pipeline
   - `data/processed/original_splits.joblib` - Train/validation/test data splits
   - `data/processed/processed_data.npz` - Preprocessed data in NumPy format

4. **Configuration**
   - `config/config.yaml` - System configuration

## Backup System

The backup system is implemented in `scripts/backup_system.py` with a convenient shell wrapper in `scripts/backup.sh`.

### Creating Backups

To create a backup:

```bash
# Create a local backup
./scripts/backup.sh backup

# Create a backup and upload to cloud storage
./scripts/backup.sh backup --cloud --storage=s3
```

When a backup is created:

1. A timestamp (YYYYMMDD_HHMMSS) is generated for the backup
2. Critical files are copied to a temporary directory
3. File checksums are computed for integrity verification
4. A manifest is created with file metadata
5. Files are compressed into a single archive (.tar.gz)
6. The archive is stored in the `backups/` directory
7. The manifest is updated with backup information
8. The temporary directory is cleaned up
9. (Optional) The archive is uploaded to cloud storage

### Listing Backups

To list available backups:

```bash
# List local backups
./scripts/backup.sh list

# List backups in cloud storage
./scripts/backup.sh list --cloud --storage=azure
```

### Restoring Backups

To restore from a backup:

```bash
# Restore from the latest local backup
./scripts/backup.sh restore

# Restore from a specific backup
./scripts/backup.sh restore --timestamp=20250331_120000

# Restore from cloud storage
./scripts/backup.sh restore --cloud --storage=s3 --timestamp=20250331_120000
```

When restoring from a backup:

1. The specified backup archive is located (or downloaded from cloud)
2. The archive is extracted to a temporary directory
3. Files are copied from the backup to their original locations
4. File integrity is verified with checksums
5. The temporary directory is cleaned up

### Pruning Old Backups

To manage storage space, old backups can be pruned:

```bash
# Keep only the 5 most recent backups (default)
./scripts/backup.sh prune

# Keep only the 3 most recent backups
./scripts/backup.sh prune --keep=3

# Prune cloud backups as well
./scripts/backup.sh prune --keep=3 --cloud --storage=s3
```

## Cloud Storage Integration

The backup system can integrate with multiple cloud storage providers for off-site backup storage.

### AWS S3

To use AWS S3 for backup storage:

1. Install the required package:
   ```bash
   pip install boto3
   ```

2. Configure credentials:
   - Copy `config/cloud_credentials.template.json` to `config/cloud_credentials.json`
   - Add your AWS credentials and S3 bucket name:
     ```json
     {
       "aws_access_key_id": "YOUR_AWS_ACCESS_KEY_ID",
       "aws_secret_access_key": "YOUR_AWS_SECRET_ACCESS_KEY",
       "s3_bucket": "your-s3-bucket-name"
     }
     ```

3. Use the `--cloud` and `--storage=s3` flags with backup commands:
   ```bash
   ./scripts/backup.sh backup --cloud --storage=s3
   ```

### Azure Blob Storage

To use Azure Blob Storage for backup storage:

1. Install the required package:
   ```bash
   pip install azure-storage-blob
   ```

2. Configure credentials:
   - Copy `config/cloud_credentials.template.json` to `config/cloud_credentials.json`
   - Add your Azure connection string and container name:
     ```json
     {
       "azure_connection_string": "YOUR_AZURE_CONNECTION_STRING",
       "azure_container": "your-azure-container-name"
     }
     ```

3. Use the `--cloud` and `--storage=azure` flags with backup commands:
   ```bash
   ./scripts/backup.sh backup --cloud --storage=azure
   ```

### Google Cloud Storage

To use Google Cloud Storage for backup storage:

1. Install the required package:
   ```bash
   pip install google-cloud-storage
   ```

2. Authenticate with GCP:
   ```bash
   gcloud auth application-default login
   ```

3. Configure credentials:
   - Copy `config/cloud_credentials.template.json` to `config/cloud_credentials.json`
   - Add your GCP bucket name:
     ```json
     {
       "gcp_bucket": "your-gcp-bucket-name"
     }
     ```

4. Use the `--cloud` and `--storage=gcp` flags with backup commands:
   ```bash
   ./scripts/backup.sh backup --cloud --storage=gcp
   ```

## Automated Backups

For production deployments, it's recommended to set up automated backups using cron or a similar scheduler.

Example crontab entry for daily backups at 2 AM:

```
0 2 * * * cd /path/to/heart-disease-mlp && ./scripts/backup.sh backup --cloud --storage=s3 >> logs/backup_cron.log 2>&1
```

Example crontab entry for weekly backup pruning (keeping the last 5 backups):

```
0 3 * * 0 cd /path/to/heart-disease-mlp && ./scripts/backup.sh prune --keep=5 --cloud --storage=s3 >> logs/backup_cron.log 2>&1
```

## Disaster Recovery

In the event of a system failure or data loss, follow these steps:

1. **Assessment**: Identify what data has been lost or corrupted.
2. **Backup Identification**: List available backups using `./scripts/backup.sh list`.
3. **Restoration**: Restore the most recent backup that predates the data loss:
   ```bash
   ./scripts/backup.sh restore --timestamp=YYYYMMDD_HHMMSS
   ```
4. **Verification**: Verify system functionality by running tests:
   ```bash
   python -m pytest tests/
   ```

## Best Practices

For optimal backup and recovery management:

1. **Regular Backups**: Schedule daily backups in production environments.
2. **Multiple Destinations**: Use both local and cloud storage for backups.
3. **Backup Testing**: Regularly test the restoration process to ensure backups are viable.
4. **Monitoring**: Check backup logs to ensure backups are completing successfully.
5. **Pruning**: Implement a retention policy to manage storage space.
6. **Documentation**: Keep a log of backup operations and any restoration events.
7. **Secure Storage**: Ensure cloud storage credentials are kept secure and never committed to version control.
8. **Version Control**: Consider using git tags to mark versions that correspond to backups.
9. **Comprehensive Testing**: After restoration, run comprehensive tests to ensure system integrity.
