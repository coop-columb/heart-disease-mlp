#!/usr/bin/env python3
"""
Scheduled backup script for Heart Disease Prediction system.

This script is intended to be run by a CI/CD system or cron job to
create regular backups of the system. It can be configured to upload
backups to cloud storage and prune old backups.

Usage:
    python scheduled_backup.py [--cloud] [--storage {s3,azure,gcp}] [--keep N]
"""

import argparse
import logging
import os
import sys

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(project_root, "logs", "scheduled_backup.log")),
    ],
)
logger = logging.getLogger("scheduled_backup")

# Make sure logs directory exists
os.makedirs(os.path.join(project_root, "logs"), exist_ok=True)


def main():
    """Run scheduled backup."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Scheduled backup for Heart Disease Prediction system"
    )
    parser.add_argument("--cloud", action="store_true", help="Upload backup to cloud storage")
    parser.add_argument("--storage", choices=["s3", "azure", "gcp"], help="Cloud storage provider")
    parser.add_argument("--keep", type=int, default=5, help="Number of backups to keep")
    args = parser.parse_args()

    logger.info("Starting scheduled backup")

    # Import backup system
    try:
        from scripts.backup_system import create_backup, prune_backups

        # Create backup
        logger.info("Creating backup...")
        timestamp = create_backup(cloud=args.cloud, storage=args.storage)
        logger.info(f"Backup created with timestamp: {timestamp}")

        # Prune old backups
        if args.keep > 0:
            logger.info(f"Pruning old backups, keeping {args.keep} most recent...")
            success = prune_backups(keep=args.keep, cloud=args.cloud, storage=args.storage)
            if success:
                logger.info("Pruning completed successfully")
            else:
                logger.error("Pruning failed")

        logger.info("Scheduled backup completed successfully")
        return 0

    except Exception as e:
        logger.error(f"Scheduled backup failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
