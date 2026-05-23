import hashlib
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from competition.config import get_vietnam_now


DEFAULT_DB_PATH = str(Path(__file__).resolve().parent.parent.parent / "competition.db")


@dataclass
class TeamRecord:
    canonical_team_id: str
    team_name: str
    primary_email: str
    status: str


@dataclass
class TeamLookupRecord:
    canonical_team_id: str
    team_name: str
    primary_email: str
    status: str
    created_at: str


@dataclass
class SubmissionRecord:
    submission_id: str
    canonical_team_id: str
    response_id: str
    drive_file_id: str
    validation_status: str
    validation_reason: Optional[str]
    extracted_path: Optional[str]
    mu: float
    sigma: float
    n_games: int
    wins: int
    draws: int
    losses: int
    total_rank: int
    total_steps: int
    is_baseline: bool
    is_active: bool
    is_team_best: bool
    is_team_recent: bool
    is_top_global: bool
    created_at: str


class SubmissionStore:
    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _connect(self):
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS teams (
                    canonical_team_id TEXT PRIMARY KEY,
                    team_name TEXT NOT NULL UNIQUE,
                    primary_email TEXT NOT NULL,
                    submission_token_hash TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'active',
                    created_at TEXT NOT NULL
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS submissions (
                    submission_id TEXT PRIMARY KEY,
                    canonical_team_id TEXT NOT NULL,
                    response_id TEXT NOT NULL UNIQUE,
                    drive_file_id TEXT NOT NULL,
                    original_filename TEXT,
                    sha256 TEXT,
                    uploaded_at TEXT,
                    created_at TEXT NOT NULL,
                    validation_status TEXT NOT NULL,
                    validation_reason TEXT,
                    extracted_path TEXT,
                    extracted_manifest_json TEXT,
                    is_baseline INTEGER NOT NULL DEFAULT 0,
                    is_active INTEGER NOT NULL DEFAULT 0,
                    is_team_best INTEGER NOT NULL DEFAULT 0,
                    is_team_recent INTEGER NOT NULL DEFAULT 0,
                    is_top_global INTEGER NOT NULL DEFAULT 0,
                    mu REAL NOT NULL DEFAULT 25.0,
                    sigma REAL NOT NULL DEFAULT 8.333,
                    n_games INTEGER NOT NULL DEFAULT 0,
                    wins INTEGER NOT NULL DEFAULT 0,
                    draws INTEGER NOT NULL DEFAULT 0,
                    losses INTEGER NOT NULL DEFAULT 0,
                    total_rank INTEGER NOT NULL DEFAULT 0,
                    total_steps INTEGER NOT NULL DEFAULT 0,
                    FOREIGN KEY (canonical_team_id) REFERENCES teams(canonical_team_id)
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS daily_submission_quota (
                    canonical_team_id TEXT NOT NULL,
                    day_utc TEXT NOT NULL,
                    submission_count INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY (canonical_team_id, day_utc),
                    FOREIGN KEY (canonical_team_id) REFERENCES teams(canonical_team_id)
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS match_results (
                    match_id TEXT PRIMARY KEY,
                    seed INTEGER,
                    player_submission_ids_csv TEXT NOT NULL,
                    ranks_csv TEXT NOT NULL,
                    steps_csv TEXT,
                    json_path TEXT,
                    gif_path TEXT,
                    json_drive_url TEXT,
                    gif_drive_url TEXT,
                    match_type TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )

            # Keep fresh installs complete while also tolerating older local DB files.
            self._ensure_submission_columns(conn)
            self._ensure_match_result_columns(conn)

            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_submissions_team_created ON submissions(canonical_team_id, created_at DESC)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_submissions_active ON submissions(is_active, validation_status)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_submissions_score ON submissions(mu, sigma, n_games)"
            )

            # Ensure a dummy team exists for logging unknown submissions
            cursor.execute(
                """
                INSERT OR IGNORE INTO teams (
                    canonical_team_id, team_name, primary_email, submission_token_hash, created_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                ("unknown_team", "unknown_team", "system@internal", "no_token", get_vietnam_now().isoformat())
            )
            conn.commit()

    def _ensure_submission_columns(self, conn: sqlite3.Connection):
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(submissions)")
        existing_cols = {row[1] for row in cursor.fetchall()}

        required_defs = {
            "is_baseline": "INTEGER NOT NULL DEFAULT 0",
            "is_active": "INTEGER NOT NULL DEFAULT 0",
            "is_team_best": "INTEGER NOT NULL DEFAULT 0",
            "is_team_recent": "INTEGER NOT NULL DEFAULT 0",
            "is_top_global": "INTEGER NOT NULL DEFAULT 0",
            "mu": "REAL NOT NULL DEFAULT 25.0",
            "sigma": "REAL NOT NULL DEFAULT 8.333",
            "n_games": "INTEGER NOT NULL DEFAULT 0",
            "wins": "INTEGER NOT NULL DEFAULT 0",
            "draws": "INTEGER NOT NULL DEFAULT 0",
            "losses": "INTEGER NOT NULL DEFAULT 0",
            "total_rank": "INTEGER NOT NULL DEFAULT 0",
            "total_steps": "INTEGER NOT NULL DEFAULT 0",
        }

        for col, definition in required_defs.items():
            if col not in existing_cols:
                cursor.execute(f"ALTER TABLE submissions ADD COLUMN {col} {definition}")

    def _ensure_match_result_columns(self, conn: sqlite3.Connection):
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(match_results)")
        existing_cols = {row[1] for row in cursor.fetchall()}

        required_defs = {
            "gif_path": "TEXT",
            "json_drive_url": "TEXT",
            "gif_drive_url": "TEXT",
        }

        for col, definition in required_defs.items():
            if col not in existing_cols:
                cursor.execute(f"ALTER TABLE match_results ADD COLUMN {col} {definition}")

    @staticmethod
    def hash_token(token: str) -> str:
        return hashlib.sha256(token.encode("utf-8")).hexdigest()

    def register_team(self, canonical_team_id: str, team_name: str, primary_email: str, token: str):
        token_hash = self.hash_token(token)
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO teams (
                    canonical_team_id,
                    team_name,
                    primary_email,
                    submission_token_hash,
                    status,
                    created_at
                )
                VALUES (?, ?, ?, ?, 'active', COALESCE(
                    (SELECT created_at FROM teams WHERE canonical_team_id = ?),
                    ?
                ))
                ON CONFLICT(canonical_team_id) DO UPDATE SET
                    team_name = excluded.team_name,
                    primary_email = excluded.primary_email,
                    submission_token_hash = excluded.submission_token_hash,
                    status = 'active'
                """,
                (
                    canonical_team_id,
                    team_name,
                    primary_email,
                    token_hash,
                    canonical_team_id,
                    get_vietnam_now().isoformat(),
                ),
            )
            conn.commit()

    def get_team(self, canonical_team_id: str) -> Optional[TeamRecord]:
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT canonical_team_id, team_name, primary_email, status
                FROM teams
                WHERE canonical_team_id = ?
                """,
                (canonical_team_id,),
            )
            row = cursor.fetchone()
            if not row:
                return None
            return TeamRecord(
                canonical_team_id=row[0],
                team_name=row[1],
                primary_email=row[2],
                status=row[3],
            )

    def get_team_by_name(self, team_name: str) -> Optional[TeamLookupRecord]:
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT canonical_team_id, team_name, primary_email, status, created_at
                FROM teams
                WHERE team_name = ?
                """,
                (team_name,),
            )
            row = cursor.fetchone()
            if not row:
                return None
            return TeamLookupRecord(
                canonical_team_id=row[0],
                team_name=row[1],
                primary_email=row[2],
                status=row[3],
                created_at=row[4],
            )

    def verify_token(self, canonical_team_id: str, token: str) -> bool:
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT submission_token_hash FROM teams WHERE canonical_team_id = ?",
                (canonical_team_id,),
            )
            row = cursor.fetchone()
            if not row:
                return False
            return row[0] == self.hash_token(token)

    def has_processed_response(self, response_id: str) -> bool:
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT 1 FROM submissions WHERE response_id = ?", (response_id,))
            return cursor.fetchone() is not None

    def increment_daily_quota(self, canonical_team_id: str, day_utc: str):
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO daily_submission_quota (canonical_team_id, day_utc, submission_count)
                VALUES (?, ?, 1)
                ON CONFLICT(canonical_team_id, day_utc) DO UPDATE SET
                    submission_count = submission_count + 1
                """,
                (canonical_team_id, day_utc),
            )
            conn.commit()

    def get_daily_quota_count(self, canonical_team_id: str, day_utc: str) -> int:
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT submission_count
                FROM daily_submission_quota
                WHERE canonical_team_id = ? AND day_utc = ?
                """,
                (canonical_team_id, day_utc),
            )
            row = cursor.fetchone()
            return int(row[0]) if row else 0

    def save_submission(
        self,
        submission_id: str,
        canonical_team_id: str,
        response_id: str,
        drive_file_id: str,
        original_filename: str,
        sha256: Optional[str],
        uploaded_at: Optional[str],
        validation_status: str,
        validation_reason: Optional[str],
        extracted_path: Optional[str],
        extracted_manifest_json: Optional[str],
    ):
        now = get_vietnam_now().isoformat()
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO submissions (
                    submission_id,
                    canonical_team_id,
                    response_id,
                    drive_file_id,
                    original_filename,
                    sha256,
                    uploaded_at,
                    created_at,
                    validation_status,
                    validation_reason,
                    extracted_path,
                    extracted_manifest_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    submission_id,
                    canonical_team_id,
                    response_id,
                    drive_file_id,
                    original_filename,
                    sha256,
                    uploaded_at,
                    now,
                    validation_status,
                    validation_reason,
                    extracted_path,
                    extracted_manifest_json,
                ),
            )
            conn.commit()

    def get_submission_by_response_id(self, response_id: str) -> Optional[SubmissionRecord]:
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT
                    submission_id,
                    canonical_team_id,
                    response_id,
                    drive_file_id,
                    validation_status,
                    validation_reason,
                    extracted_path,
                    mu,
                    sigma,
                    n_games,
                    wins,
                    draws,
                    losses,
                    total_rank,
                    total_steps,
                    is_baseline,
                    is_active,
                    is_team_best,
                    is_team_recent,
                    is_top_global,
                    created_at
                FROM submissions
                WHERE response_id = ?
                """,
                (response_id,),
            )
            row = cursor.fetchone()
            if not row:
                return None
            return SubmissionRecord(
                submission_id=row[0],
                canonical_team_id=row[1],
                response_id=row[2],
                drive_file_id=row[3],
                validation_status=row[4],
                validation_reason=row[5],
                extracted_path=row[6],
                mu=float(row[7]),
                sigma=float(row[8]),
                n_games=int(row[9]),
                wins=int(row[10]),
                draws=int(row[11]),
                losses=int(row[12]),
                total_rank=int(row[13]),
                total_steps=int(row[14]),
                is_baseline=bool(row[15]),
                is_active=bool(row[16]),
                is_team_best=bool(row[17]),
                is_team_recent=bool(row[18]),
                is_top_global=bool(row[19]),
                created_at=row[20],
            )

    def update_submission_pool_flags(
        self,
        submission_id: str,
        *,
        is_baseline: bool,
        is_active: bool,
        is_team_best: bool,
        is_team_recent: bool,
        is_top_global: bool,
    ):
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE submissions
                SET
                    is_baseline = ?,
                    is_active = ?,
                    is_team_best = ?,
                    is_team_recent = ?,
                    is_top_global = ?
                WHERE submission_id = ?
                """,
                (
                    int(is_baseline),
                    int(is_active),
                    int(is_team_best),
                    int(is_team_recent),
                    int(is_top_global),
                    submission_id,
                ),
            )
            conn.commit()

    def list_valid_submissions(self) -> list[SubmissionRecord]:
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT
                    submission_id,
                    canonical_team_id,
                    response_id,
                    drive_file_id,
                    validation_status,
                    validation_reason,
                    extracted_path,
                    mu,
                    sigma,
                    n_games,
                    wins,
                    draws,
                    losses,
                    total_rank,
                    total_steps,
                    is_baseline,
                    is_active,
                    is_team_best,
                    is_team_recent,
                    is_top_global,
                    created_at
                FROM submissions
                WHERE validation_status = 'valid' AND extracted_path IS NOT NULL
                ORDER BY created_at DESC
                """
            )
            rows = cursor.fetchall()
            return [
                SubmissionRecord(
                    submission_id=row[0],
                    canonical_team_id=row[1],
                    response_id=row[2],
                    drive_file_id=row[3],
                    validation_status=row[4],
                    validation_reason=row[5],
                    extracted_path=row[6],
                    mu=float(row[7]),
                    sigma=float(row[8]),
                    n_games=int(row[9]),
                    wins=int(row[10]),
                    draws=int(row[11]),
                    losses=int(row[12]),
                    total_rank=int(row[13]),
                    total_steps=int(row[14]),
                    is_baseline=bool(row[15]),
                    is_active=bool(row[16]),
                    is_team_best=bool(row[17]),
                    is_team_recent=bool(row[18]),
                    is_top_global=bool(row[19]),
                    created_at=row[20],
                )
                for row in rows
            ]

    def list_feedback_submissions(self) -> list[dict]:
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT
                    s.submission_id,
                    s.canonical_team_id,
                    t.team_name,
                    s.validation_status,
                    s.validation_reason,
                    s.created_at
                FROM submissions s
                JOIN teams t ON t.canonical_team_id = s.canonical_team_id
                ORDER BY s.created_at DESC
                """
            )
            rows = cursor.fetchall()

        return [
            {
                "submission_id": row[0],
                "canonical_team_id": row[1],
                "team_name": row[2],
                "validation_status": row[3],
                "validation_reason": row[4],
                "created_at": row[5],
            }
            for row in rows
        ]

    def list_match_results(self) -> list[dict]:
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT
                    match_id,
                    seed,
                    player_submission_ids_csv,
                    ranks_csv,
                    steps_csv,
                    json_path,
                    gif_path,
                    json_drive_url,
                    gif_drive_url,
                    match_type,
                    created_at
                FROM match_results
                ORDER BY created_at DESC
                """
            )
            rows = cursor.fetchall()

        return [
            {
                "match_id": row[0],
                "seed": row[1],
                "player_submission_ids_csv": row[2],
                "ranks_csv": row[3],
                "steps_csv": row[4],
                "json_path": row[5],
                "gif_path": row[6],
                "json_drive_url": row[7],
                "gif_drive_url": row[8],
                "match_type": row[9],
                "created_at": row[10],
            }
            for row in rows
        ]

    def save_match_result(
        self,
        match_id: str,
        seed: Optional[int],
        player_submission_ids_csv: str,
        ranks_csv: str,
        steps_csv: Optional[str],
        json_path: Optional[str],
        gif_path: Optional[str] = None,
        json_drive_url: Optional[str] = None,
        gif_drive_url: Optional[str] = None,
        match_type: str = "submission_batch",
    ):
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO match_results (
                    match_id,
                    seed,
                    player_submission_ids_csv,
                    ranks_csv,
                    steps_csv,
                    json_path,
                    gif_path,
                    json_drive_url,
                    gif_drive_url,
                    match_type,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    match_id,
                    seed,
                    player_submission_ids_csv,
                    ranks_csv,
                    steps_csv,
                    json_path,
                    gif_path,
                    json_drive_url,
                    gif_drive_url,
                    match_type,
                    get_vietnam_now().isoformat(),
                ),
            )
            conn.commit()

    def mark_submission_runtime_invalid(self, submission_id: str, reason: str):
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE submissions
                SET
                    validation_status = 'invalid',
                    validation_reason = ?,
                    is_active = 0,
                    is_team_best = 0,
                    is_team_recent = 0,
                    is_top_global = 0
                WHERE submission_id = ?
                """,
                (reason, submission_id),
            )
            conn.commit()
