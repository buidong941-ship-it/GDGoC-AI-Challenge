"""Ingestion module: submission intake and validation logic."""

from .collector import (
    authenticate_submission,
    download_drive_file_bytes,
    extract_zip_bytes,
    get_drive_service,
    load_submission_metadata,
    process_submission_item,
    validate_zip_bytes,
    now_iso,
    MAX_ZIP_SIZE_BYTES,
    MAX_EXTRACTED_TOTAL_BYTES,
    MAX_SINGLE_FILE_BYTES,
    MAX_FILE_COUNT,
    ALLOWED_EXTENSIONS,
)

__all__ = [
    "authenticate_submission",
    "download_drive_file_bytes",
    "extract_zip_bytes",
    "get_drive_service",
    "load_submission_metadata",
    "process_submission_item",
    "validate_zip_bytes",
    "now_iso",
    "MAX_ZIP_SIZE_BYTES",
    "MAX_EXTRACTED_TOTAL_BYTES",
    "MAX_SINGLE_FILE_BYTES",
    "MAX_FILE_COUNT",
    "ALLOWED_EXTENSIONS",
]
