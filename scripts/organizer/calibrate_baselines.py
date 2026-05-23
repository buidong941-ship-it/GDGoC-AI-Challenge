import argparse
import hashlib
import json
import logging
import os
import random
import sqlite3
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from competition.config import load_env
load_env()

from competition.storage import SubmissionStore
from competition.evaluation.match_runner import MatchRunner
from competition.evaluation.ranking import RankingSystem

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parent.parent.parent

BASELINES = [
    {
        "name": "tactical_rule_agent",
        "team_id": "baseline_tactical_rule_agent",
        "submission_id": "baseline_tactical_rule_agent_v1",
        "response_id": "baseline:tactical_rule_agent:v1",
        "drive_file_id": "baseline:tactical_rule_agent:v1",
        "agent_path": ROOT_DIR / "submissions" / "baseline_tactical_rule_agent" / "baseline_tactical_rule_agent_v1" / "agent.py",
    },
    {
        "name": "genius_rule_agent",
        "team_id": "baseline_genius_rule_agent",
        "submission_id": "baseline_genius_rule_agent_v1",
        "response_id": "baseline:genius_rule_agent:v1",
        "drive_file_id": "baseline:genius_rule_agent:v1",
        "agent_path": ROOT_DIR / "submissions" / "baseline_genius_rule_agent" / "baseline_genius_rule_agent_v1" / "agent.py",
    },
    {
        "name": "smarter_rule_agent",
        "team_id": "baseline_smarter_rule_agent",
        "submission_id": "baseline_smarter_rule_agent_v1",
        "response_id": "baseline:smarter_rule_agent:v1",
        "drive_file_id": "baseline:smarter_rule_agent:v1",
        "agent_path": ROOT_DIR / "submissions" / "baseline_smarter_rule_agent" / "baseline_smarter_rule_agent_v1" / "agent.py",
    },
    {
        "name": "box_farmer_agent",
        "team_id": "baseline_box_farmer_agent",
        "submission_id": "baseline_box_farmer_agent_v1",
        "response_id": "baseline:box_farmer_agent:v1",
        "drive_file_id": "baseline:box_farmer_agent:v1",
        "agent_path": ROOT_DIR / "submissions" / "baseline_box_farmer_agent" / "baseline_box_farmer_agent_v1" / "agent.py",
    },
    {
        "name": "simple_rule_agent",
        "team_id": "baseline_simple_rule_agent",
        "submission_id": "baseline_simple_rule_agent_v1",
        "response_id": "baseline:simple_rule_agent:v1",
        "drive_file_id": "baseline:simple_rule_agent:v1",
        "agent_path": ROOT_DIR / "submissions" / "baseline_simple_rule_agent" / "baseline_simple_rule_agent_v1" / "agent.py",
    },
    {
        "name": "random_agent",
        "team_id": "baseline_random_agent",
        "submission_id": "baseline_random_agent_v1",
        "response_id": "baseline:random_agent:v1",
        "drive_file_id": "baseline:random_agent:v1",
        "agent_path": ROOT_DIR / "submissions" / "baseline_random_agent" / "baseline_random_agent_v1" / "agent.py",
    },
]

def reset_baselines_to_default(db_path: str):
    logger.info("Resetting baseline ratings to mu=100.0, sigma=33.33, n_games=0...")
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()

        for baseline in BASELINES:
            agent_path = baseline["agent_path"]
            if not agent_path.exists():
                raise FileNotFoundError(f"Missing baseline agent.py: {agent_path}")

            extracted_path = str(agent_path.parent)
            manifest = {"agent.py": agent_path.stat().st_size}

            token_hash = hashlib.sha256(
                f"{baseline['team_id']}_token".encode("utf-8")
            ).hexdigest()
            
            cursor.execute(
                """
                INSERT INTO teams (
                    canonical_team_id, team_name, primary_email, submission_token_hash, status, created_at
                ) VALUES (?, ?, ?, ?, 'active', ?)
                ON CONFLICT(canonical_team_id) DO UPDATE SET
                    team_name = excluded.team_name,
                    primary_email = excluded.primary_email,
                    submission_token_hash = excluded.submission_token_hash,
                    status = 'active'
                """,
                (
                    baseline["team_id"],
                    baseline["team_id"],
                    "baseline@local",
                    token_hash,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )

            cursor.execute(
                """
                INSERT INTO submissions (
                    submission_id, canonical_team_id, response_id, drive_file_id, original_filename,
                    created_at, validation_status, extracted_path, extracted_manifest_json,
                    is_baseline, is_active, is_team_best, is_team_recent, is_top_global,
                    mu, sigma, n_games, wins, draws, losses, total_rank, total_steps
                ) VALUES (?, ?, ?, ?, ?, datetime('now'), 'valid', ?, ?, 1, 1, 1, 1, 1, 100.0, ?, 0, 0, 0, 0, 0, 0)
                ON CONFLICT(submission_id) DO UPDATE SET
                    validation_status = 'valid',
                    extracted_path = excluded.extracted_path,
                    extracted_manifest_json = excluded.extracted_manifest_json,
                    is_baseline = 1, is_active = 1, is_team_best = 1, is_team_recent = 1, is_top_global = 1,
                    mu = 100.0, sigma = ?, n_games = 0, wins = 0, draws = 0, losses = 0, total_rank = 0, total_steps = 0
                """,
                (
                    baseline["submission_id"], baseline["team_id"], baseline["response_id"], baseline["drive_file_id"],
                    f"{baseline['name']}.py", extracted_path, json.dumps(manifest, sort_keys=True),
                    100/3, 100/3
                ),
            )

        conn.commit()
    logger.info("Baselines successfully reset.")


def run_single_calibration_match(
    match_index: int,
    participants: list[str],
    db_path: str,
    enable_gif: bool = False,
    timeout_s: float = 15.0
):
    agent_paths = []
    team_ids = []
    
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        for sid in participants:
            cursor.execute(
                """
                SELECT s.extracted_path, t.team_name 
                FROM submissions s
                JOIN teams t ON s.canonical_team_id = t.canonical_team_id
                WHERE s.submission_id = ?
                """,
                (sid,)
            )
            row = cursor.fetchone()
            if row:
                agent_path = os.path.join(row[0], "agent.py")
                agent_paths.append(agent_path)
                team_ids.append(row[1])
            else:
                return {"status": "error", "reason": f"submission {sid} not found"}

    # Temporarily disable Drive uploads for calibration matches to save API quota and Drive space
    if "DRIVE_FOLDER_ID" in os.environ:
        os.environ["DRIVE_FOLDER_ID"] = ""

    runner = MatchRunner(
        log_dir="logs",
        enable_gif=enable_gif,
    )
    
    try:
        ranks, survival_steps, gif_path, json_path, gif_drive_url, json_drive_url = runner.run_match(
            agent_paths=agent_paths,
            team_ids=participants, # Keep submission IDs for ranking system mapping
            seed=random.randint(0, 999999),
            max_steps=500,
            inference_timeout_s=0.1,
            startup_timeout_s=timeout_s
        )
        return {
            "status": "success",
            "participants": participants,
            "ranks": ranks,
            "steps": survival_steps,
            "json_path": json_path,
            "gif_path": gif_path,
        }
    except Exception as e:
        return {
            "status": "error",
            "participants": participants,
            "reason": str(e)
        }

def calibrate_baselines(db_path: str, matches: int, parallel_workers: int, skip_reset: bool = False):
    if not skip_reset:
        reset_baselines_to_default(db_path)

    # Fetch baselines
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT submission_id FROM submissions WHERE is_baseline = 1 AND validation_status = 'valid'")
        baseline_ids = [row[0] for row in cursor.fetchall()]

    if len(baseline_ids) < 4:
        logger.error(f"Need at least 4 valid baselines to calibrate. Found {len(baseline_ids)}.")
        return

    logger.info(f"Found {len(baseline_ids)} baselines. Starting {matches} calibration matches using {parallel_workers} workers.")
    
    match_queue = []
    for _ in range(matches):
        match_queue.append(random.sample(baseline_ids, 4))
        
    ranking = RankingSystem(db_path=db_path)
    
    success_count = 0
    error_count = 0
    
    start_time = time.time()
    
    with ProcessPoolExecutor(max_workers=parallel_workers) as executor:
        futures = {
            executor.submit(run_single_calibration_match, i, p, db_path): p 
            for i, p in enumerate(match_queue)
        }
        
        for future in as_completed(futures):
            res = future.result()
            if res["status"] == "success":
                success_count += 1
                # The critical part: update ratings AND allow baselines to be updated
                ranking.update_ratings(
                    submission_ids=res["participants"],
                    ranks=res["ranks"],
                    steps=res["steps"],
                    json_path=res["json_path"],
                    gif_path=res["gif_path"],
                    match_type="baseline_calibration",
                    allow_baseline_updates=True
                )
            else:
                error_count += 1
                logger.error(f"Calibration match failed: {res.get('reason')}")
            
            completed = success_count + error_count
            if completed % 10 == 0 or completed == matches:
                logger.info(f"Progress: {completed}/{matches} matches complete")
                
    duration = time.time() - start_time
    logger.info(f"Calibration complete in {duration:.1f}s. Success: {success_count}, Errors: {error_count}")
    
    # Print updated baseline stats
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT t.team_name, s.mu, s.sigma, s.n_games 
            FROM submissions s 
            JOIN teams t ON s.canonical_team_id = t.canonical_team_id 
            WHERE s.is_baseline = 1 
            ORDER BY s.mu DESC
            """
        )
        print("\n--- Final Baseline Ratings ---")
        for row in cursor.fetchall():
            print(f"{row[0]:<35} Mu: {row[1]:.2f} | Sigma: {row[2]:.2f} | Games: {row[3]}")

    # Update Google Sheets Leaderboard
    try:
        from competition.integrations.notifications import update_google_sheets
        logger.info("Updating Google Sheet Leaderboard...")
        sheet_result = update_google_sheets(
            db_path=db_path,
            credentials_file=None,
            spreadsheet_id=None,
            sheet_range=None,
            include_baseline=True,
        )
        logger.info(f"Sheet update result: {sheet_result}")
    except Exception as e:
        logger.error(f"Failed to update Google Sheet: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reset and calibrate baseline ratings.")
    parser.add_argument("--matches", type=int, default=600, help="Number of calibration matches to run")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--db_path", type=str, default="competition.db", help="Path to DB")
    parser.add_argument("--skip_reset", action="store_true", help="Skip resetting baseline scores to 100 before running matches")
    args = parser.parse_args()
    
    calibrate_baselines(args.db_path, args.matches, args.workers, args.skip_reset)
