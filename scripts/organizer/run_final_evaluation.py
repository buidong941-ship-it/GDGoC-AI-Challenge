import argparse
import logging
import random
import time
import os
import sqlite3
import json
import itertools
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent

from competition.config import load_env
load_env()

from competition.evaluation.match_runner import MatchRunner

DEFAULT_DB_PATH = str(ROOT_DIR / "competition.db")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Map rank (0=best, 3=worst) to tournament points
RANK_TO_POINTS = {
    0: 3,
    1: 2,
    2: 1,
    3: 0
}

def run_single_finals_match(
    match_index: int,
    participants: list[str],
    db_path: str,
    enable_gif: bool = False,
    timeout_s: float = 15.0
):
    agent_paths = []
    team_names = []
    
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
                team_names.append(row[1])
            else:
                return {"status": "error", "reason": f"submission {sid} not found"}

    log_dir = os.path.join("logs", "finals")
    os.makedirs(log_dir, exist_ok=True)
    
    runner = MatchRunner(
        log_dir=log_dir,
        enable_gif=enable_gif,
    )
    
    try:
        ranks, survival_steps, gif_path, json_path, _, _ = runner.run_match(
            agent_paths=agent_paths,
            team_ids=team_names, # Use team names for nicer GIF rendering
            seed=random.randint(0, 999999),
            max_steps=500,
            inference_timeout_s=0.1,
            startup_timeout_s=timeout_s
        )
        
        # Calculate points based on returned ranks
        points = [RANK_TO_POINTS[r] for r in ranks]
        
        return {
            "status": "success",
            "participants": participants,
            "team_names": team_names,
            "ranks": ranks,
            "points": points,
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

def run_final_evaluation(db_path: str, matches_per_combo: int, parallel_workers: int, enable_gif: bool):
    logger.info("--- FREEZING LEADERBOARD AND STARTING GRAND FINALS ---")
    
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        
        # Get Top 8 Student Teams by strict score -> mu -> sigma tiebreaking
        cursor.execute(
            """
            SELECT s.submission_id, t.team_name, s.mu, s.sigma, (s.mu - 3 * s.sigma) as score
            FROM submissions s
            JOIN teams t ON s.canonical_team_id = t.canonical_team_id
            WHERE s.is_baseline = 0 
              AND s.validation_status = 'valid'
              AND s.is_team_best = 1
            ORDER BY score DESC, s.mu DESC, s.sigma ASC
            LIMIT 8
            """
        )
        student_rows = cursor.fetchall()
        
        # Get Top 1 Baseline Agent by strict score -> mu -> sigma tiebreaking
        cursor.execute(
            """
            SELECT s.submission_id, t.team_name, s.mu, s.sigma, (s.mu - 3 * s.sigma) as score
            FROM submissions s
            JOIN teams t ON s.canonical_team_id = t.canonical_team_id
            WHERE s.is_baseline = 1 
              AND s.validation_status = 'valid'
            ORDER BY score DESC, s.mu DESC, s.sigma ASC
            LIMIT 1
            """
        )
        baseline_rows = cursor.fetchall()

    roster = student_rows + baseline_rows
    if len(roster) < 4:
        logger.error(f"Cannot run finals. Need at least 4 participants, found {len(roster)}.")
        return

    logger.info(f"Finals Roster:")
    
    # Store initial metrics for tie-breaking
    initial_metrics = {}
    for sid, tname, mu, sigma, score in roster:
        initial_metrics[sid] = {"mu": mu, "sigma": sigma, "score": score}
        logger.info(f"  - {tname:<30} (Score: {score:.2f} | Mu: {mu:.2f} | Sigma: {sigma:.2f})")
        
    participant_ids = [r[0] for r in roster]
    id_to_name = {r[0]: r[1] for r in roster}
    
    # Generate all unique combinations
    combos = list(itertools.combinations(participant_ids, 4))
    logger.info(f"Generated {len(combos)} unique 4-player combinations.")
    
    match_queue = []
    for combo in combos:
        for _ in range(matches_per_combo):
            # Shuffle so they start in random corner positions
            shuffled_combo = list(combo)
            random.shuffle(shuffled_combo)
            match_queue.append(shuffled_combo)
            
    total_matches = len(match_queue)
    logger.info(f"Queued {total_matches} total matches ({matches_per_combo} per combination).")
    
    success_count = 0
    error_count = 0
    
    # Score tracking
    scores = {sid: 0 for sid in participant_ids}
    games_played = {sid: 0 for sid in participant_ids}
    
    start_time = time.time()
    
    with ProcessPoolExecutor(max_workers=parallel_workers) as executor:
        futures = {
            executor.submit(run_single_finals_match, i, p, db_path, enable_gif): p 
            for i, p in enumerate(match_queue)
        }
        
        for future in as_completed(futures):
            res = future.result()
            if res["status"] == "success":
                success_count += 1
                for i, sid in enumerate(res["participants"]):
                    scores[sid] += res["points"][i]
                    games_played[sid] += 1
            else:
                error_count += 1
                logger.error(f"Match failed: {res.get('reason')}")
                
            completed = success_count + error_count
            if completed % 50 == 0 or completed == total_matches:
                logger.info(f"Progress: {completed}/{total_matches} matches complete")
                
    duration = time.time() - start_time
    logger.info(f"Finals complete in {duration:.1f}s. Success: {success_count}, Errors: {error_count}")
    
    # Calculate final standings
    standings = []
    for sid in participant_ids:
        standings.append({
            "team_name": id_to_name[sid],
            "submission_id": sid,
            "points": scores[sid],
            "matches_played": games_played[sid],
            "avg_points_per_match": scores[sid] / max(1, games_played[sid]),
            "initial_score": initial_metrics[sid]["score"],
            "initial_mu": initial_metrics[sid]["mu"],
            "initial_sigma": initial_metrics[sid]["sigma"]
        })
        
    # Sort by points descending, then by score -> mu -> sigma (ascending for sigma)
    standings.sort(
        key=lambda x: (
            x["points"], 
            x["initial_score"], 
            x["initial_mu"], 
            -x["initial_sigma"]
        ), 
        reverse=True
    )
    
    print("\\n=======================================================")
    print("                 GRAND FINALS LEADERBOARD              ")
    print("=======================================================")
    for rank, st in enumerate(standings, 1):
        print(f"{rank}. {st['team_name']:<35} | Points: {st['points']:<5} | Matches: {st['matches_played']}")
    print("=======================================================\\n")
    
    # Save JSON report
    report = {
        "timestamp": datetime.now().isoformat(),
        "total_matches_run": success_count,
        "matches_failed": error_count,
        "standings": standings
    }
    
    os.makedirs(os.path.join("logs", "finals"), exist_ok=True)
    report_path = os.path.join("logs", "finals", "finals_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)
        
    logger.info(f"Saved detailed report to {report_path}")

    # Push to Google Sheets "Grand Final" tab
    credentials_file = os.getenv("LEADERBOARD_CREDENTIALS_FILE", "secrets/service_account_credentials.json")
    spreadsheet_id = os.getenv("LEADERBOARD_SPREADSHEET_ID")
    
    if spreadsheet_id and os.path.exists(credentials_file):
        try:
            from google.oauth2 import service_account
            from googleapiclient.discovery import build
            from competition.integrations.notifications import _ensure_sheet_tab, _write_sheet_values
            
            creds = service_account.Credentials.from_service_account_file(
                credentials_file, scopes=["https://www.googleapis.com/auth/spreadsheets"]
            )
            service = build("sheets", "v4", credentials=creds)
            
            sheet_name = "Grand Final"
            _ensure_sheet_tab(service, spreadsheet_id, sheet_name)
            
            values = [["Rank", "Team Name", "Submission ID", "Total Points", "Matches Played", "Avg Points/Match"]]
            for rank, st in enumerate(standings, 1):
                values.append([
                    str(rank),
                    st["team_name"],
                    st["submission_id"],
                    str(st["points"]),
                    str(st["matches_played"]),
                    f"{st['avg_points_per_match']:.4f}"
                ])
                
            _write_sheet_values(service, spreadsheet_id, sheet_name, values)
            logger.info(f"Successfully pushed Grand Finals leaderboard to Google Sheets tab '{sheet_name}'.")
        except Exception as e:
            logger.error(f"Failed to push to Google Sheets: {e}")
    else:
        logger.warning("Spreadsheet ID or Credentials not found. Skipping Google Sheets update.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the isolated Grand Finals tournament.")
    parser.add_argument("--matches_per_combo", type=int, default=50, help="Number of matches to run for each 4-player combination")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--db_path", type=str, default=DEFAULT_DB_PATH, help="Path to DB")
    parser.add_argument("--enable_gif", action="store_true", help="Generate GIFs for finals matches")
    if hasattr(os, "geteuid") and os.geteuid() != 0:
        print("ERROR: For security, evaluation must be run with sudo to enable sandboxing.")
        import sys
        sys.exit(1)
        
    args = parser.parse_args()
    
    run_final_evaluation(args.db_path, args.matches_per_combo, args.workers, args.enable_gif)
