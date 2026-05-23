import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Optional

import trueskill

from competition.storage import SubmissionStore


class RankingSystem:
    def __init__(self, db_path="competition.db"):
        self.db_path = db_path
        self.env = trueskill.TrueSkill(mu=100.0, sigma=100/3, draw_probability=0.1)
        # Ensures all canonical tables/columns exist in a schema-first way.
        self.store = SubmissionStore(db_path=db_path)

    @staticmethod
    def score(mu: float, sigma: float) -> float:
        return float(mu) - 3.0 * float(sigma)

    def get_submission_rating(self, submission_id: str) -> trueskill.Rating:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT mu, sigma FROM submissions WHERE submission_id = ?",
                (submission_id,),
            )
            row = cursor.fetchone()
            if row:
                return self.env.Rating(mu=float(row[0]), sigma=float(row[1]))
            return self.env.Rating()

    def get_baseline_flags(self, submission_ids: list[str]) -> dict[str, bool]:
        if not submission_ids:
            return {}

        placeholders = ",".join("?" for _ in submission_ids)
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"""
                SELECT submission_id, is_baseline
                FROM submissions
                WHERE submission_id IN ({placeholders})
                """,
                tuple(submission_ids),
            )
            rows = cursor.fetchall()
        return {row[0]: bool(row[1]) for row in rows}

    def update_ratings(
        self,
        submission_ids: list[str],
        ranks: list[int],
        steps: Optional[list[int]] = None,
        seed: Optional[int] = None,
        json_path: Optional[str] = None,
        gif_path: Optional[str] = None,
        json_drive_url: Optional[str] = None,
        gif_drive_url: Optional[str] = None,
        match_type: str = "submission_batch",
        allow_baseline_updates: bool = False,
    ):
        if len(submission_ids) != len(ranks):
            raise ValueError("submission_ids and ranks must have the same length")
        if steps is not None and len(steps) != len(submission_ids):
            raise ValueError("steps must be None or have same length as submission_ids")

        ratings = [self.get_submission_rating(sid) for sid in submission_ids]
        rating_groups = [(r,) for r in ratings]
        new_groups = self.env.rate(rating_groups, ranks=ranks)
        baseline_flags = self.get_baseline_flags(submission_ids)

        min_rank = min(ranks)
        winners = {i for i, rank in enumerate(ranks) if rank == min_rank}
        multi_winner = len(winners) > 1

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            for i, sid in enumerate(submission_ids):
                if not allow_baseline_updates and baseline_flags.get(sid, False):
                    continue

                new_mu = float(new_groups[i][0].mu)
                new_sigma = float(new_groups[i][0].sigma)

                win_delta = 1 if (i in winners and not multi_winner) else 0
                draw_delta = 1 if (i in winners and multi_winner) else 0
                loss_delta = 1 if i not in winners else 0
                step_delta = int(steps[i]) if steps is not None else 0

                cursor.execute(
                    """
                    UPDATE submissions
                    SET
                        mu = ?,
                        sigma = ?,
                        n_games = n_games + 1,
                        wins = wins + ?,
                        draws = draws + ?,
                        losses = losses + ?,
                        total_rank = total_rank + ?,
                        total_steps = total_steps + ?
                    WHERE submission_id = ?
                    """,
                    (
                        new_mu,
                        new_sigma,
                        win_delta,
                        draw_delta,
                        loss_delta,
                        int(ranks[i]),
                        step_delta,
                        sid,
                    ),
                )

            conn.commit()

        self.store.save_match_result(
            match_id=str(uuid.uuid4()),
            seed=seed,
            player_submission_ids_csv=",".join(submission_ids),
            ranks_csv=",".join(str(r) for r in ranks),
            steps_csv=",".join(str(s) for s in steps) if steps is not None else None,
            json_path=json_path,
            gif_path=gif_path,
            json_drive_url=json_drive_url,
            gif_drive_url=gif_drive_url,
            match_type=match_type,
        )

    def get_leaderboard(self, include_baseline: bool = True):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            baseline_clause = "" if include_baseline else "AND s.is_baseline = 0"
            cursor.execute(
                f"""
                SELECT
                    s.submission_id,
                    s.canonical_team_id,
                    t.team_name,
                    s.mu,
                    s.sigma,
                    s.n_games,
                    s.wins,
                    s.draws,
                    s.losses,
                    s.total_rank,
                    s.total_steps,
                    s.is_baseline,
                    s.is_active,
                    s.created_at
                FROM submissions s
                JOIN teams t ON t.canonical_team_id = s.canonical_team_id
                WHERE s.validation_status = 'valid' {baseline_clause}
                ORDER BY (s.mu - 3 * s.sigma) DESC, s.mu DESC, s.sigma ASC, s.created_at DESC
                """
            )
            rows = cursor.fetchall()

        leaderboard = []
        for row in rows:
            avg_rank = (row[9] / row[5]) if row[5] > 0 else 0.0
            avg_steps = (row[10] / row[5]) if row[5] > 0 else 0.0
            win_rate = (row[6] / row[5]) if row[5] > 0 else 0.0
            leaderboard.append(
                {
                    "submission_id": row[0],
                    "canonical_team_id": row[1],
                    "team_name": row[2],
                    "mu": float(row[3]),
                    "sigma": float(row[4]),
                    "score": self.score(row[3], row[4]),
                    "n_games": int(row[5]),
                    "wins": int(row[6]),
                    "draws": int(row[7]),
                    "losses": int(row[8]),
                    "total_rank": int(row[9]),
                    "total_steps": int(row[10]),
                    "win_rate": win_rate,
                    "avg_rank": avg_rank,
                    "avg_steps": avg_steps,
                    "is_baseline": bool(row[11]),
                    "is_active": bool(row[12]),
                    "created_at": row[13],
                }
            )
        return leaderboard


if __name__ == "__main__":
    rs = RankingSystem(db_path="competition.db")
    rows = rs.get_leaderboard()
    for i, row in enumerate(rows[:10], start=1):
        print(
            f"{i}. {row['team_name']} [{row['submission_id']}] "
            f"score={row['score']:.2f} mu={row['mu']:.2f} sigma={row['sigma']:.2f} games={row['n_games']}"
        )
