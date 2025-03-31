#!/bin/bash

# Backup script for Heart Disease Prediction system
# This script provides a convenient wrapper around the backup_system.py module

# Ensure script is run from the project root
cd "$(dirname "$0")/.." || exit 1

# Create backup directory if it doesn't exist
mkdir -p backups

# Make sure the backup script is executable
chmod +x scripts/backup_system.py

# Function to display usage
function display_usage {
    echo "Heart Disease Prediction Backup and Recovery System"
    echo ""
    echo "Usage: $0 [command] [options]"
    echo ""
    echo "Commands:"
    echo "  backup   Create a new backup"
    echo "  restore  Restore from a backup"
    echo "  list     List available backups"
    echo "  prune    Prune old backups"
    echo ""
    echo "Options:"
    echo "  --cloud              Use cloud storage"
    echo "  --storage=PROVIDER   Cloud storage provider (s3, azure, gcp)"
    echo "  --timestamp=TIME     Backup timestamp for restore (default: latest)"
    echo "  --keep=N             Number of backups to keep when pruning (default: 5)"
    echo ""
    echo "Examples:"
    echo "  $0 backup                   Create a local backup"
    echo "  $0 backup --cloud --storage=s3  Create a backup and upload to AWS S3"
    echo "  $0 list                     List local backups"
    echo "  $0 list --cloud --storage=azure  List backups in Azure Blob Storage"
    echo "  $0 restore                  Restore from the latest local backup"
    echo "  $0 restore --timestamp=20250331_120000  Restore from specific backup"
    echo "  $0 prune --keep=3          Keep only the 3 most recent backups"
    echo ""
}

# Check if command was provided
if [ $# -eq 0 ]; then
    display_usage
    exit 1
fi

# Extract command (first argument)
COMMAND=$1
shift

# Process options
CLOUD=""
STORAGE=""
TIMESTAMP=""
KEEP=""

for arg in "$@"; do
    case $arg in
        --cloud)
            CLOUD="--cloud"
            ;;
        --storage=*)
            STORAGE="--storage ${arg#*=}"
            ;;
        --timestamp=*)
            TIMESTAMP="--timestamp ${arg#*=}"
            ;;
        --keep=*)
            KEEP="--keep ${arg#*=}"
            ;;
        --help)
            display_usage
            exit 0
            ;;
        *)
            echo "Unknown option: $arg"
            display_usage
            exit 1
            ;;
    esac
done

# Run the backup system with the provided command and options
case $COMMAND in
    backup)
        echo "Creating backup..."
        python scripts/backup_system.py backup $CLOUD $STORAGE
        ;;
    restore)
        echo "Restoring from backup..."
        python scripts/backup_system.py restore $TIMESTAMP $CLOUD $STORAGE
        ;;
    list)
        echo "Listing backups..."
        python scripts/backup_system.py list $CLOUD $STORAGE
        ;;
    prune)
        echo "Pruning old backups..."
        python scripts/backup_system.py prune $KEEP $CLOUD $STORAGE
        ;;
    *)
        echo "Unknown command: $COMMAND"
        display_usage
        exit 1
        ;;
esac

exit $?