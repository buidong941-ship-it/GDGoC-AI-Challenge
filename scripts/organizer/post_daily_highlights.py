import argparse
import os
import requests
import re
from datetime import timedelta
import sqlite3

from competition.config import get_vietnam_now
from competition.storage import SubmissionStore
from competition.evaluation.ranking import RankingSystem

def extract_drive_id(url: str) -> str:
    # Match various Google Drive URL formats
    match = re.search(r'/d/([a-zA-Z0-9_-]+)/', url)
    if match:
        return match.group(1)
    match = re.search(r'id=([a-zA-Z0-9_-]+)', url)
    if match:
        return match.group(1)
    return ""

def post_highlights(db_path: str = "competition.db"):
    webhook_url = os.getenv("DISCORD_HIGHLIGHTS_WEBHOOK_URL", "")
    if not webhook_url:
        print("Missing DISCORD_HIGHLIGHTS_WEBHOOK_URL. Exiting.")
        return

    # 1. Get Top 5 Leaderboard
    ranking = RankingSystem(db_path=db_path)
    leaderboard = ranking.get_leaderboard(include_baseline=True)
    
    # 2. Find Highlight of the Day (Most Wins in the last 24h)
    now = get_vietnam_now()
    yesterday_str = (now - timedelta(days=1)).isoformat()
    
    store = SubmissionStore(db_path=db_path)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT player_submission_ids_csv, ranks_csv, gif_drive_url 
        FROM match_results 
        WHERE created_at >= ?
        """, 
        (yesterday_str,)
    )
    matches_today = cursor.fetchall()
    
    wins_counter = {}
    best_match_for_sub = {}
    
    for (subs_csv, ranks_csv, gif_url) in matches_today:
        subs = subs_csv.split(',')
        ranks = [int(r) for r in ranks_csv.split(',')]
        
        # Find solo winners
        min_rank = min(ranks)
        winners = [i for i, r in enumerate(ranks) if r == min_rank]
        
        if len(winners) == 1:
            winner_idx = winners[0]
            sub_id = subs[winner_idx]
            wins_counter[sub_id] = wins_counter.get(sub_id, 0) + 1
            if gif_url and sub_id not in best_match_for_sub:
                best_match_for_sub[sub_id] = gif_url

    # Exclude baselines from highlight if possible
    valid_subs = store.list_valid_submissions()
    baseline_ids = {s.submission_id for s in valid_subs if s.is_baseline}
    
    best_sub_id = None
    max_wins = -1
    for sub_id, wins in wins_counter.items():
        if sub_id not in baseline_ids and wins > max_wins:
            max_wins = wins
            best_sub_id = sub_id
            
    # Fallback to baselines if no user agents won
    if not best_sub_id and wins_counter:
        for sub_id, wins in wins_counter.items():
            if wins > max_wins:
                max_wins = wins
                best_sub_id = sub_id

    embeds = []
    
    # Highlight Embed
    if best_sub_id:
        cursor.execute(
            """
            SELECT t.team_name, s.mu, s.n_games 
            FROM submissions s
            JOIN teams t ON s.canonical_team_id = t.canonical_team_id
            WHERE s.submission_id = ?
            """, 
            (best_sub_id,)
        )
        team_row = cursor.fetchone()
        
        if team_row:
            team_name, mu, n_games = team_row
            gif_url = best_match_for_sub.get(best_sub_id)
            
            highlight_embed = {
                "title": "🏆 Highlight of the Day!",
                "description": f"**Team:** {team_name}\n**Submission ID:** `{best_sub_id}`\n**Wins Today:** {max_wins}\n**Current Mu:** {mu:.2f} ({n_games} total games)",
                "color": 16766720, # Gold
            }
            
            content = f"🔥 Today's top performer was **{team_name}** with {max_wins} wins! 🔥\n"
            
            # Since Discord doesn't natively autoplay Google Drive GIFs, we provide a clickable link
            if gif_url:
                content += f"Watch the winning match: {gif_url}\n"
            
            embeds.append(highlight_embed)
        else:
            content = "Here is your daily update!"
    else:
        content = "No matches were played today. Here is the current leaderboard!"

    # Leaderboard Embed
    if leaderboard:
        lb_embed = {
            "title": "📊 Current Leaderboard (Top 5)", 
            "color": 3447003, # Blue
            "fields": []
        }
        for i, row in enumerate(leaderboard[:5], start=1):
            lb_embed["fields"].append(
                {
                    "name": f"#{i} {row['team_name']}",
                    "value": f"Score: {row['score']:.2f} | mu={row['mu']:.2f} | games={row['n_games']}",
                    "inline": False,
                }
            )
        embeds.append(lb_embed)

    payload = {"content": content, "embeds": embeds}

    try:
        r = requests.post(webhook_url, json=payload, timeout=10)
        r.raise_for_status()
        print("Successfully posted daily highlights to Discord.")
    except Exception as e:
        print(f"Failed to post to Discord: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Post Daily Highlights to Discord")
    parser.add_argument("--db_path", type=str, default="competition.db", help="Path to SQLite DB")
    args = parser.parse_args()
    
    post_highlights(args.db_path)
