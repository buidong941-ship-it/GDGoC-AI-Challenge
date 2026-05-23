import mimetypes
import os
import fcntl
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Optional

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow


JSON_MIME_TYPE = "application/json"
GIF_MIME_TYPE = "image/gif"
DEFAULT_SCOPES = ["https://www.googleapis.com/auth/drive"]
DEFAULT_CLIENT_SECRETS_FILE = str(Path(__file__).resolve().parent / "client_secrets.json")
DEFAULT_TOKEN_FILE = str(Path(__file__).resolve().parent / "token.json")
DEFAULT_LOCK_DIR = Path(__file__).resolve().parent / ".drive_locks"


def _mime_type_for_path(local_path: str) -> str:
    suffix = Path(local_path).suffix.lower()
    if suffix == ".json":
        return JSON_MIME_TYPE
    if suffix == ".gif":
        return GIF_MIME_TYPE
    mime_type, _ = mimetypes.guess_type(local_path)
    return mime_type or "application/octet-stream"


def _save_user_credentials(credentials: Credentials, token_path: str) -> None:
    token_file = Path(token_path)
    token_file.parent.mkdir(parents=True, exist_ok=True)
    token_file.write_text(credentials.to_json())


@lru_cache(maxsize=8)
def _build_drive_service(client_secrets_path: str, token_path: str):
    token_file = Path(token_path)
    creds: Optional[Credentials] = None

    if token_file.exists():
        creds = Credentials.from_authorized_user_file(str(token_file), scopes=DEFAULT_SCOPES)

    if creds and creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
            _save_user_credentials(creds, token_path)
        except Exception as e:
            # If refresh fails (e.g. revoked or expired testing token), delete the token file
            # to force a re-authentication next time.
            if token_file.exists():
                token_file.unlink()
            raise RuntimeError(
                f"Google Drive token refresh failed: {e}. "
                "The token has been deleted. Please run `python -m competition.integrations.drive_oauth` to re-authorize."
            ) from e

    if not creds or not creds.valid:
        raise RuntimeError(
            "Drive OAuth token not found or invalid. Run `python -m evaluation.drive_oauth` once to authorize the user account."
        )

    return build("drive", "v3", credentials=creds)


def get_drive_service(client_secrets_path: Optional[str] = None, token_path: Optional[str] = None):
    client_secrets_path = client_secrets_path or os.getenv("DRIVE_OAUTH_CLIENT_SECRETS") or DEFAULT_CLIENT_SECRETS_FILE
    token_path = token_path or os.getenv("DRIVE_OAUTH_TOKEN_FILE") or DEFAULT_TOKEN_FILE
    return _build_drive_service(str(client_secrets_path), str(token_path))


def create_drive_token(client_secrets_path: Optional[str] = None, token_path: Optional[str] = None) -> str:
    client_secrets_path = client_secrets_path or os.getenv("DRIVE_OAUTH_CLIENT_SECRETS") or DEFAULT_CLIENT_SECRETS_FILE
    token_path = token_path or os.getenv("DRIVE_OAUTH_TOKEN_FILE") or DEFAULT_TOKEN_FILE

    flow = InstalledAppFlow.from_client_secrets_file(str(client_secrets_path), scopes=DEFAULT_SCOPES)
    creds = flow.run_local_server(port=0)
    _save_user_credentials(creds, str(token_path))
    return str(token_path)


def _find_child_folder_id(service, parent_folder_id: str, folder_name: str) -> Optional[str]:
    query = (
        "mimeType='application/vnd.google-apps.folder' and "
        f"name='{folder_name}' and '{parent_folder_id}' in parents and trashed=false"
    )
    response = (
        service.files()
        .list(q=query, fields="files(id, name)", pageSize=10, supportsAllDrives=True, includeItemsFromAllDrives=True)
        .execute()
    )
    files = response.get("files", [])
    return files[0]["id"] if files else None


def _create_folder(service, parent_folder_id: str, folder_name: str) -> str:
    metadata = {
        "name": folder_name,
        "mimeType": "application/vnd.google-apps.folder",
        "parents": [parent_folder_id],
    }
    folder = (
        service.files()
        .create(body=metadata, fields="id", supportsAllDrives=True)
        .execute()
    )
    return folder["id"]


def ensure_drive_folder(service, parent_folder_id: str, folder_name: str) -> str:
    folder_id = _find_child_folder_id(service, parent_folder_id, folder_name)
    if folder_id:
        return folder_id
    return _create_folder(service, parent_folder_id, folder_name)


def _folder_lock_path(parent_folder_id: str, folder_name: str) -> Path:
    safe_name = folder_name.replace(os.sep, "_")
    return DEFAULT_LOCK_DIR / f"{parent_folder_id}_{safe_name}.lock"


def _ensure_drive_folder_locked(service, parent_folder_id: str, folder_name: str) -> str:
    DEFAULT_LOCK_DIR.mkdir(parents=True, exist_ok=True)
    lock_path = _folder_lock_path(parent_folder_id, folder_name)
    with open(lock_path, "w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            return ensure_drive_folder(service, parent_folder_id, folder_name)
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)


def _drive_view_url(file_id: str) -> str:
    return f"https://drive.google.com/file/d/{file_id}/view?usp=drive_link"


def upload_file_to_drive(client_secrets_path: Optional[str], folder_id: str, local_path: str) -> dict:
    """Upload a local file to Drive under folder_id/json-or-gifs/YYYY-MM-DD.

    The root `folder_id` is treated as the organizer-shared parent folder.
    Files are placed into `json/` or `gifs/` subfolders, then a UTC date folder.
    """
    if not folder_id:
        raise ValueError("folder_id is required for Drive upload")

    local_path = str(local_path)
    service = get_drive_service(client_secrets_path=client_secrets_path)
    local_file = Path(local_path)
    if not local_file.exists():
        raise FileNotFoundError(local_path)

    artifact_folder = "gifs" if local_file.suffix.lower() == ".gif" else "json"
    date_folder = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    artifact_root_id = _ensure_drive_folder_locked(service, folder_id, artifact_folder)
    date_folder_id = _ensure_drive_folder_locked(service, artifact_root_id, date_folder)

    media = {
        "mimeType": _mime_type_for_path(local_path),
        "body": open(local_path, "rb"),
    }
    metadata = {
        "name": local_file.name,
        "parents": [date_folder_id],
    }

    try:
        from googleapiclient.http import MediaIoBaseUpload

        with open(local_path, "rb") as fh:
            media = MediaIoBaseUpload(fh, mimetype=_mime_type_for_path(local_path), resumable=True)
            file = (
                service.files()
                .create(
                    body=metadata,
                    media_body=media,
                    fields="id, webViewLink",
                    supportsAllDrives=True,
                )
                .execute()
            )
    except Exception:
        raise

    file_id = file["id"]
    return {
        "file_id": file_id,
        "web_view_link": file.get("webViewLink") or _drive_view_url(file_id),
        "artifact_folder": artifact_folder,
        "date_folder": date_folder,
    }
