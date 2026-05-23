"""Canonical team ID generation helpers."""

import re
import uuid


_TEAM_SLUG_PATTERN = re.compile(r"[^a-z0-9]+")


def _slugify_team_name(team_name: str) -> str:
    slug = _TEAM_SLUG_PATTERN.sub("_", team_name.strip().lower()).strip("_")
    return slug or "team"


def generate_canonical_team_id(team_name: str, suffix_length: int = 8) -> str:
    """Generate a collision-resistant canonical team ID.

    Format: {slugified_team_name}_{random_suffix}
    Example: "AI Warriors" -> "ai_warriors_a1b2c3d4"
    """
    if not team_name or not team_name.strip():
        raise ValueError("team_name must be a non-empty string")
    if suffix_length < 4:
        raise ValueError("suffix_length must be at least 4")

    slug = _slugify_team_name(team_name)
    suffix = uuid.uuid4().hex[:suffix_length]
    return f"{slug}_{suffix}"
