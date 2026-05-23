"""Destructively reset competition DB to baseline-only state.

Usage examples:
    python -m evaluation.reset_to_baselines --db_path competition.db --dry_run
    python -m evaluation.reset_to_baselines --db_path competition.db --yes
"""

from __future__ import annotations

import argparse
import sqlite3
from dataclasses import dataclass
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent.parent

from competition.config import load_env
load_env()
DEFAULT_DB_PATH = str(ROOT_DIR / "competition.db")


@dataclass
class ResetStats:
    baseline_submissions: int
    nonbaseline_submissions: int
    match_rows_total: int
    match_rows_to_delete: int
    teams_total: int
    teams_to_delete: int
    quota_rows_to_delete: int


def _csv_ids(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item and item.strip()]


def _collect_state(conn: sqlite3.Connection, delete_empty_teams: bool) -> tuple[ResetStats, list[str], list[str]]:
    cursor = conn.cursor()

    cursor.execute("SELECT submission_id FROM submissions WHERE is_baseline = 1")
    baseline_submission_ids = [row[0] for row in cursor.fetchall()]
    baseline_set = set(baseline_submission_ids)

    cursor.execute("SELECT submission_id FROM submissions WHERE is_baseline = 0")
    nonbaseline_submission_ids = [row[0] for row in cursor.fetchall()]
    nonbaseline_set = set(nonbaseline_submission_ids)

    cursor.execute("SELECT match_id FROM match_results")
    match_rows = cursor.fetchall()
    match_ids_to_delete: list[str] = [row[0] for row in match_rows]


    team_ids_to_delete: list[str] = []
    quota_rows_to_delete = 0

    cursor.execute("SELECT COUNT(*) FROM teams")
    teams_total = int(cursor.fetchone()[0])

    if delete_empty_teams:
        cursor.execute(
            """
            SELECT t.canonical_team_id
            FROM teams t
            LEFT JOIN submissions s
              ON s.canonical_team_id = t.canonical_team_id
             AND s.is_baseline = 1
            WHERE s.submission_id IS NULL
            """
        )
        team_ids_to_delete = [row[0] for row in cursor.fetchall()]
        if team_ids_to_delete:
            placeholders = ",".join("?" for _ in team_ids_to_delete)
            cursor.execute(
                f"SELECT COUNT(*) FROM daily_submission_quota WHERE canonical_team_id IN ({placeholders})",
                team_ids_to_delete,
            )
            quota_rows_to_delete = int(cursor.fetchone()[0])

    stats = ResetStats(
        baseline_submissions=len(baseline_submission_ids),
        nonbaseline_submissions=len(nonbaseline_submission_ids),
        match_rows_total=len(match_rows),
        match_rows_to_delete=len(match_ids_to_delete),
        teams_total=teams_total,
        teams_to_delete=len(team_ids_to_delete),
        quota_rows_to_delete=quota_rows_to_delete,
    )

    return stats, match_ids_to_delete, team_ids_to_delete


def reset_to_baselines(
    db_path: str,
    *,
    dry_run: bool,
    delete_empty_teams: bool,
    allow_empty_baseline: bool,
) -> ResetStats:
    with sqlite3.connect(db_path) as conn:
        stats, match_ids_to_delete, team_ids_to_delete = _collect_state(
            conn, delete_empty_teams=delete_empty_teams
        )

        if stats.baseline_submissions == 0 and not allow_empty_baseline:
            raise RuntimeError(
                "No baseline submissions found. Refusing reset. "
                "Use --allow_empty_baseline to override."
            )

        if dry_run:
            return stats

        cursor = conn.cursor()
        cursor.execute("BEGIN")
        try:
            if match_ids_to_delete:
                placeholders = ",".join("?" for _ in match_ids_to_delete)
                cursor.execute(
                    f"DELETE FROM match_results WHERE match_id IN ({placeholders})",
                    match_ids_to_delete,
                )

            cursor.execute("DELETE FROM submissions WHERE is_baseline = 0")

            if delete_empty_teams and team_ids_to_delete:
                placeholders = ",".join("?" for _ in team_ids_to_delete)
                cursor.execute(
                    f"DELETE FROM daily_submission_quota WHERE canonical_team_id IN ({placeholders})",
                    team_ids_to_delete,
                )
                cursor.execute(
                    f"DELETE FROM teams WHERE canonical_team_id IN ({placeholders})",
                    team_ids_to_delete,
                )

            conn.commit()
        except Exception:
            conn.rollback()
            raise

    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM submissions WHERE is_baseline = 0")
        remaining_nonbaseline = int(cursor.fetchone()[0])
        if remaining_nonbaseline != 0:
            raise RuntimeError("Reset verification failed: non-baseline submissions still exist.")

    return stats


def _print_stats(stats: ResetStats, *, dry_run: bool, delete_empty_teams: bool):
    mode = "DRY RUN" if dry_run else "APPLIED"
    print(f"[{mode}] Baseline-only reset summary")
    print(f"- baseline submissions kept: {stats.baseline_submissions}")
    print(f"- non-baseline submissions removed: {stats.nonbaseline_submissions}")
    print(f"- match_results total scanned: {stats.match_rows_total}")
    print(f"- match_results removed: {stats.match_rows_to_delete}")
    if delete_empty_teams:
        print(f"- teams removed (no baseline submission): {stats.teams_to_delete}")
        print(f"- daily quota rows removed: {stats.quota_rows_to_delete}")
    else:
        print("- teams kept: enabled via --keep_teams")


def main():
    parser = argparse.ArgumentParser(description="Reset competition DB to baseline-only rows")
    parser.add_argument("--db_path", default=DEFAULT_DB_PATH, help="Path to SQLite DB")
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Show what would be deleted without modifying the DB",
    )
    parser.add_argument(
        "--keep_teams",
        action="store_true",
        help="Keep team and daily quota rows even if no baseline submission remains",
    )
    parser.add_argument(
        "--allow_empty_baseline",
        action="store_true",
        help="Allow reset to continue even if no baseline submissions are found",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Confirm destructive execution (required unless --dry_run is used)",
    )
    parser.add_argument(
        "--update_sheet",
        action="store_true",
        help="Explicitly refresh Google Sheet after reset (default behavior for applied resets)",
    )
    parser.add_argument(
        "--skip_sheet_update",
        action="store_true",
        help="Skip Google Sheet refresh after reset",
    )
    parser.add_argument("--sheet_credentials_file", default=None)
    parser.add_argument("--sheet_spreadsheet_id", default=None)
    parser.add_argument("--sheet_range", default=None)
    args = parser.parse_args()

    if not args.dry_run and not args.yes:
        raise SystemExit("Refusing destructive reset without --yes. Use --dry_run to preview.")

    stats = reset_to_baselines(
        db_path=args.db_path,
        dry_run=args.dry_run,
        delete_empty_teams=not args.keep_teams,
        allow_empty_baseline=args.allow_empty_baseline,
    )
    _print_stats(stats, dry_run=args.dry_run, delete_empty_teams=not args.keep_teams)

    should_update_sheet = (not args.skip_sheet_update) and (not args.dry_run)
    if args.update_sheet and args.skip_sheet_update:
        raise SystemExit("Choose only one of --update_sheet or --skip_sheet_update.")

    if should_update_sheet:
        try:
            from competition.integrations.notifications import update_google_sheets

            sheet_result = update_google_sheets(
                db_path=args.db_path,
                credentials_file=args.sheet_credentials_file,
                spreadsheet_id=args.sheet_spreadsheet_id,
                sheet_range=args.sheet_range,
                include_baseline=True,
            )
            print(f"- sheet_update: {sheet_result}")
        except Exception as e:
            print(f"- sheet_update: error:{e}")
    elif args.update_sheet and not args.dry_run:
        try:
            from competition.integrations.notifications import update_google_sheets

            sheet_result = update_google_sheets(
                db_path=args.db_path,
                credentials_file=args.sheet_credentials_file,
                spreadsheet_id=args.sheet_spreadsheet_id,
                sheet_range=args.sheet_range,
                include_baseline=True,
            )
            print(f"- sheet_update: {sheet_result}")
        except Exception as e:
            print(f"- sheet_update: error:{e}")


if __name__ == "__main__":
    main()
