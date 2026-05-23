"""
CLI wrapper for submission collection and team management.

This module provides the command-line interface for the submission intake system.
Core logic is delegated to competition.ingestion and competition.storage modules.
"""

import argparse
from pathlib import Path

from competition.ingestion import get_drive_service, load_submission_metadata, process_submission_item
from competition.storage import SubmissionStore

# Configuration
ROOT_DIR = Path(__file__).resolve().parent.parent.parent

from competition.config import load_env
load_env()
DEFAULT_DB_PATH = str(ROOT_DIR / "competition.db")
DEFAULT_STORAGE_DIR = str(ROOT_DIR / "submissions")
DEFAULT_CREDENTIALS_FILE = "secrets/service_account_credentials.json"
DEFAULT_METADATA_FILE = str(ROOT_DIR / "evaluation" / "submission_metadata.json")


def run_collection(credentials_file: str, metadata_file: str, db_path: str, storage_dir: str):
    """
    Orchestrate submission collection from Google Drive.

    Args:
        credentials_file: Path to service account credentials JSON
        metadata_file: Path to submission metadata JSON (list of dicts)
        db_path: Path to SQLite database
        storage_dir: Root directory for extracted submissions

    Loads metadata entries, iterates through each, and processes via competition.ingestion.
    """
    store = SubmissionStore(db_path)
    service = get_drive_service(credentials_file)
    if not service:
        print("Google Drive service not available. Check credentials.")
        return

    metadata_rows = load_submission_metadata(metadata_file)
    if not metadata_rows:
        print("No metadata rows found.")
        return

    print(f"Processing {len(metadata_rows)} metadata entries...")
    for item in metadata_rows:
        drive_file_id = item.get("drive_file_id", "<missing>")
        ok, note = process_submission_item(service, store, storage_dir, item)
        status = "OK" if ok else "FAILED"
        print(f"[{status}] drive_file_id={drive_file_id} -> {note}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Bomberland submission collection and team management"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_db_cmd = subparsers.add_parser("init-db", help="Initialize SQLite database")
    init_db_cmd.add_argument("--db-path", default=DEFAULT_DB_PATH, help="Path to database file")

    upsert_team_cmd = subparsers.add_parser(
        "upsert-team", help="Register or update a team with submission token"
    )
    upsert_team_cmd.add_argument("--db-path", default=DEFAULT_DB_PATH, help="Path to database file")
    upsert_team_cmd.add_argument("--team-id", required=True, help="Canonical team ID (immutable)")
    upsert_team_cmd.add_argument("--team-name", required=True, help="Team display name (unique)")
    upsert_team_cmd.add_argument("--primary-email", default="", help="Team contact email")
    upsert_team_cmd.add_argument("--token", required=True, help="Reusable submission token")

    collect_cmd = subparsers.add_parser("collect", help="Download and process submissions from Drive")
    collect_cmd.add_argument(
        "--credentials-file",
        default=DEFAULT_CREDENTIALS_FILE,
        help="Path to service account credentials JSON",
    )
    collect_cmd.add_argument(
        "--metadata-file",
        default=DEFAULT_METADATA_FILE,
        help="Path to submission metadata JSON (list of items)",
    )
    collect_cmd.add_argument("--db-path", default=DEFAULT_DB_PATH, help="Path to database file")
    collect_cmd.add_argument(
        "--storage-dir", default=DEFAULT_STORAGE_DIR, help="Root directory for extracted submissions"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.command == "init-db":
        store = SubmissionStore(args.db_path)
        print(f"Initialized database: {args.db_path}")
    elif args.command == "upsert-team":
        store = SubmissionStore(args.db_path)
        store.register_team(args.team_id, args.team_name, args.primary_email, args.token)
        print(f"Upserted team: {args.team_id}")
    elif args.command == "collect":
        run_collection(args.credentials_file, args.metadata_file, args.db_path, args.storage_dir)
