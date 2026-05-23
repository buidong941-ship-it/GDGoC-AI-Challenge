"""Submission token generation helpers."""

import os


def generate_submission_token(num_bytes: int = 32) -> str:
    """Generate a secure reusable submission token.

    Default uses 32 random bytes -> 64 hex chars.
    """
    if num_bytes < 16:
        raise ValueError("num_bytes must be at least 16")
    return os.urandom(num_bytes).hex()
