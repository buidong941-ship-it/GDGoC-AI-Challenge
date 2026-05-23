"""
Backup competition.db to Google Drive with a timestamped filename.

Usage:
    python3 -m scripts.backup_db
    python3 -m scripts.backup_db --db_path competition.db

The backup is uploaded to the DRIVE_FOLDER_ID folder under a 'backups/' subfolder.
"""

import argparse
import logging
import os
import shutil
from datetime import datetime
from pathlib import Path

from competition.config import load_env
load_env()

from competition.integrations.drive_upload import (
    get_drive_service,
    ensure_drive_folder,
    _ensure_drive_folder_locked,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def backup_database(db_path: str):
    db_file = Path(db_path)
    if not db_file.exists():
        logger.error(f"Database file not found: {db_path}")
        return

    # Create a local timestamped copy first (atomic snapshot)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = Path("backups")
    backup_dir.mkdir(exist_ok=True)
    backup_name = f"competition_backup_{timestamp}.db"
    backup_path = backup_dir / backup_name

    shutil.copy2(db_path, backup_path)
    logger.info(f"Local backup created: {backup_path}")

    # Upload to Google Drive
    folder_id = os.getenv("DRIVE_FOLDER_ID")
    if not folder_id:
        logger.warning("DRIVE_FOLDER_ID not set. Skipping Drive upload (local backup retained).")
        return

    try:
        service = get_drive_service()
        backups_folder_id = _ensure_drive_folder_locked(service, folder_id, "backups")

        from googleapiclient.http import MediaFileUpload

        media = MediaFileUpload(str(backup_path), mimetype="application/x-sqlite3", resumable=True)
        metadata = {
            "name": backup_name,
            "parents": [backups_folder_id],
        }
        uploaded = (
            service.files()
            .create(body=metadata, media_body=media, fields="id, webViewLink", supportsAllDrives=True)
            .execute()
        )

        file_id = uploaded["id"]
        link = uploaded.get("webViewLink", f"https://drive.google.com/file/d/{file_id}/view")
        logger.info(f"Backup uploaded to Google Drive: {link}")

        # Clean up local backup after successful upload
        backup_path.unlink()
        logger.info("Local backup file removed after successful upload.")

    except Exception as e:
        logger.error(f"Drive upload failed (local backup retained at {backup_path}): {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backup competition.db to Google Drive.")
    parser.add_argument("--db_path", type=str, default="competition.db", help="Path to the database file")
    args = parser.parse_args()

    backup_database(args.db_path)
