import sqlite3

from competition.storage import SubmissionStore


class PoolManager:
    def __init__(self, db_path: str = "competition.db"):
        self.db_path = db_path
        self.store = SubmissionStore(db_path=db_path)

    def recompute_active_pool(self, recent_per_team: int = 2, top_k: int = 10) -> dict:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                UPDATE submissions
                SET
                    is_active = 0,
                    is_team_best = 0,
                    is_team_recent = 0,
                    is_top_global = 0
                """
            )

            # Best submission per team by score.
            cursor.execute(
                """
                WITH ranked AS (
                    SELECT
                        submission_id,
                        canonical_team_id,
                        ROW_NUMBER() OVER (
                            PARTITION BY canonical_team_id
                            ORDER BY (mu - 3 * sigma) DESC, created_at DESC
                        ) AS rn
                    FROM submissions
                    WHERE validation_status = 'valid'
                )
                UPDATE submissions
                SET is_team_best = 1
                WHERE submission_id IN (
                    SELECT submission_id FROM ranked WHERE rn = 1
                )
                """
            )

            # Recent submissions per team.
            cursor.execute(
                """
                WITH ranked AS (
                    SELECT
                        submission_id,
                        canonical_team_id,
                        ROW_NUMBER() OVER (
                            PARTITION BY canonical_team_id
                            ORDER BY created_at DESC
                        ) AS rn
                    FROM submissions
                    WHERE validation_status = 'valid'
                )
                UPDATE submissions
                SET is_team_recent = 1
                WHERE submission_id IN (
                    SELECT submission_id FROM ranked WHERE rn <= ?
                )
                """,
                (recent_per_team,),
            )

            # Top global submissions among sufficiently rated records.
            cursor.execute(
                """
                UPDATE submissions
                SET is_top_global = 1
                WHERE submission_id IN (
                    SELECT submission_id
                    FROM submissions
                    WHERE validation_status = 'valid' AND n_games >= 10
                    ORDER BY (mu - 3 * sigma) DESC, created_at DESC
                    LIMIT ?
                )
                """,
                (top_k,),
            )

            cursor.execute(
                """
                UPDATE submissions
                SET is_active = CASE
                    WHEN validation_status != 'valid' THEN 0
                    WHEN is_baseline = 1 THEN 1
                    WHEN is_team_best = 1 THEN 1
                    WHEN is_team_recent = 1 THEN 1
                    WHEN is_top_global = 1 THEN 1
                    ELSE 0
                END
                """
            )

            conn.commit()

            cursor.execute("SELECT COUNT(*) FROM submissions WHERE validation_status = 'valid'")
            total_valid = int(cursor.fetchone()[0])
            cursor.execute("SELECT COUNT(*) FROM submissions WHERE is_active = 1")
            total_active = int(cursor.fetchone()[0])
            cursor.execute("SELECT COUNT(*) FROM submissions WHERE is_team_best = 1")
            total_best = int(cursor.fetchone()[0])
            cursor.execute("SELECT COUNT(*) FROM submissions WHERE is_team_recent = 1")
            total_recent = int(cursor.fetchone()[0])
            cursor.execute("SELECT COUNT(*) FROM submissions WHERE is_top_global = 1")
            total_top = int(cursor.fetchone()[0])

        return {
            "total_valid": total_valid,
            "total_active": total_active,
            "total_team_best": total_best,
            "total_team_recent": total_recent,
            "total_top_global": total_top,
        }
