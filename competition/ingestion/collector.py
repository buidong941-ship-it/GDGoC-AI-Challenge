"""Ingestion collector: core submission intake logic independent of CLI."""

import hashlib
import io
import json
import uuid
import zipfile
from datetime import datetime
from pathlib import Path, PurePosixPath

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

from competition.storage import SubmissionStore

# Validation constants
MAX_ZIP_SIZE_BYTES = 100 * 1024 * 1024
MAX_EXTRACTED_TOTAL_BYTES = 300 * 1024 * 1024
MAX_SINGLE_FILE_BYTES = 150 * 1024 * 1024
MAX_FILE_COUNT = 20

ALLOWED_EXTENSIONS = {
    ".py", ".txt", ".pt", ".pth", ".pkl", ".onnx", ".bin", ".json", ".yaml", ".yml", ".md",
    ".h5", ".pb", ".keras", ".tflite"
}


def now_iso() -> str:
    """ISO format timestamp with 'Z' suffix (UTC)."""
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def get_drive_service(credentials_file: str):
    """Create Google Drive API service from service account credentials."""
    import os
    
    if not os.path.exists(credentials_file):
        print(f"Error: credentials file not found: {credentials_file}")
        return None
    creds = service_account.Credentials.from_service_account_file(
        credentials_file,
        scopes=["https://www.googleapis.com/auth/drive.readonly"],
    )
    return build("drive", "v3", credentials=creds)


def load_submission_metadata(metadata_file: str):
    """Load JSON metadata list with drive_file_id, canonical_team_id, submission_token."""
    import os
    
    if not os.path.exists(metadata_file):
        raise FileNotFoundError(
            f"Metadata file not found: {metadata_file}. "
            "Expected JSON list with: drive_file_id, canonical_team_id, submission_token, original_filename(optional)."
        )
    with open(metadata_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Metadata file must be a JSON array.")
    return data


def download_drive_file_bytes(service, file_id: str) -> bytes:
    """Download file bytes from Google Drive given a file ID."""
    request = service.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    return fh.getvalue()


def _is_path_safe(path_str: str) -> bool:
    """Check if zip member path is safe (no absolute paths, no .. traversal)."""
    p = PurePosixPath(path_str)
    if p.is_absolute():
        return False
    if any(part == ".." for part in p.parts):
        return False
    return True


def validate_zip_bytes(zip_data: bytes):
    """
    Validate zip archive for safety and structure.
    
    Returns: (valid: bool, reason: Optional[str], manifest: Optional[dict])
    - valid=True: manifest contains {filename: file_size} for extracted files
    - valid=False: reason describes validation failure, manifest is None
    """
    if len(zip_data) > MAX_ZIP_SIZE_BYTES:
        return False, "zip_too_large", None

    try:
        zf = zipfile.ZipFile(io.BytesIO(zip_data), "r")
    except Exception as e:
        return False, f"invalid_zip:{e}", None

    infos = zf.infolist()
    if len(infos) > MAX_FILE_COUNT:
        return False, "too_many_files", None

    extracted_total = 0
    agent_candidates = []
    manifest = {}

    for info in infos:
        name = info.filename
        if info.is_dir():
            continue
        if not _is_path_safe(name):
            return False, "unsafe_path", None

        suffix = Path(name).suffix.lower()
        if suffix and suffix not in ALLOWED_EXTENSIONS:
            return False, f"disallowed_extension:{suffix}", None

        if info.file_size > MAX_SINGLE_FILE_BYTES:
            return False, "single_file_too_large", None

        extracted_total += info.file_size
        if extracted_total > MAX_EXTRACTED_TOTAL_BYTES:
            return False, "extracted_total_too_large", None

        manifest[name] = info.file_size
        if name == "agent.py":
            agent_candidates.append(name)

    if len(agent_candidates) != 1:
        return False, "agent_py_must_be_at_root_of_zip", None

    # Forbid requirements.txt as the environment is fixed
    if any(Path(name).name == "requirements.txt" for name in manifest.keys()):
        return False, "requirements_txt_forbidden", None

    # Syntax sanity check for agent.py
    try:
        agent_src = zf.read(agent_candidates[0]).decode("utf-8")
        compile(agent_src, agent_candidates[0], "exec")
    except Exception as e:
        return False, f"agent_py_syntax_error:{e}", None

    return True, None, manifest


def extract_zip_bytes(zip_data: bytes, target_dir: Path, manifest: dict) -> None:
    import os
    """Extract zip archive to target directory based on manifest."""
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # Ensure all parent directories up to 'submissions' are accessible to 'nobody'
    current = target_dir
    while current.name:
        try:
            os.chmod(current, 0o755)
        except Exception:
            pass
        if current.name == 'submissions':
            break
        current = current.parent
    
    with zipfile.ZipFile(io.BytesIO(zip_data), "r") as zf:
        for rel_name in manifest.keys():
            destination = target_dir / rel_name
            destination.parent.mkdir(parents=True, exist_ok=True)
            
            # Chmod all intermediate directories inside the zip
            curr_dest = destination.parent
            while curr_dest != target_dir:
                try:
                    os.chmod(curr_dest, 0o755)
                except Exception:
                    pass
                curr_dest = curr_dest.parent
            
            with zf.open(rel_name) as source, open(destination, "wb") as dst:
                dst.write(source.read())
            
            # Ensure the extracted file is readable by the sandboxed 'nobody' user
            os.chmod(destination, 0o644)


def authenticate_submission(store: SubmissionStore, canonical_team_id: str, submission_token: str):
    """
    Verify team identity and token.
    
    Returns: (ok: bool, reason: Optional[str])
    - ok=True: authentication successful, reason is None
    - ok=False: reason describes auth failure (unknown_team, team_not_active, token_mismatch)
    """
    team = store.get_team(canonical_team_id)
    if team is None:
        return False, "unknown_team"
    if team.status != "active":
        return False, f"team_not_active:{team.status}"
    if not store.verify_token(canonical_team_id, submission_token):
        return False, "token_mismatch"
    return True, None


def process_submission_item(
    service,
    store: SubmissionStore,
    storage_dir: str,
    item: dict,
):
    """
    Orchestrate full submission intake: auth → download → validate → extract → save.
    
    Args:
        service: Google Drive API service
        store: SubmissionStore instance for DB operations
        storage_dir: Root directory for extracted submissions (e.g., "submissions/")
        item: Metadata dict with drive_file_id, canonical_team_id, submission_token, optional original_filename
    
    Returns: (ok: bool, note: str)
    - ok=True: submission successfully stored or already processed
    - ok=False: submission rejected or failed; note describes reason
    """
    required = ["drive_file_id", "canonical_team_id", "submission_token"]
    missing = [k for k in required if not item.get(k)]
    if missing:
        return False, f"missing_metadata_fields:{','.join(missing)}"

    drive_file_id = item["drive_file_id"]
    canonical_team_id = item["canonical_team_id"]
    submission_token = item["submission_token"]
    original_filename = item.get("original_filename")

    if store.has_processed_response(drive_file_id):
        return True, "already_processed"

    ok, auth_reason = authenticate_submission(store, canonical_team_id, submission_token)
    if not ok:
        # Record failed auth but don't save submission record (trust issue)
        return False, f"auth_failed:{auth_reason}"

    try:
        zip_data = download_drive_file_bytes(service, drive_file_id)
    except Exception as e:
        return False, f"download_failed:{e}"

    sha = hashlib.sha256(zip_data).hexdigest()
    valid, reason, manifest = validate_zip_bytes(zip_data)
    submission_id = str(uuid.uuid4())

    if not valid:
        store.save_submission(
            submission_id=submission_id,
            canonical_team_id=canonical_team_id,
            response_id=drive_file_id,
            drive_file_id=drive_file_id,
            original_filename=original_filename,
            sha256=sha,
            uploaded_at=now_iso(),
            validation_status="invalid",
            validation_reason=reason,
            extracted_path=None,
            extracted_manifest_json=None,
        )
        return False, reason

    target_dir = Path(storage_dir) / canonical_team_id / submission_id
    try:
        extract_zip_bytes(zip_data, target_dir, manifest)
    except Exception as e:
        store.save_submission(
            submission_id=submission_id,
            canonical_team_id=canonical_team_id,
            response_id=drive_file_id,
            drive_file_id=drive_file_id,
            original_filename=original_filename,
            sha256=sha,
            uploaded_at=now_iso(),
            validation_status="invalid",
            validation_reason=f"extract_failed:{e}",
            extracted_path=None,
            extracted_manifest_json=json.dumps(manifest or {}, sort_keys=True),
        )
        return False, f"extract_failed:{e}"

    store.save_submission(
        submission_id=submission_id,
        canonical_team_id=canonical_team_id,
        response_id=drive_file_id,
        drive_file_id=drive_file_id,
        original_filename=original_filename,
        sha256=sha,
        uploaded_at=now_iso(),
        validation_status="valid",
        validation_reason=None,
        extracted_path=str(target_dir),
        extracted_manifest_json=json.dumps(manifest, sort_keys=True),
    )
    return True, f"stored:{target_dir}"
