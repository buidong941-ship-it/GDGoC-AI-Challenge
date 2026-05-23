"""Registration automation package."""

from .canonical_id_generator import generate_canonical_team_id
from .token_generator import generate_submission_token
from .webhook_receiver import process_registration_payload

__all__ = [
    "generate_canonical_team_id",
    "generate_submission_token",
    "process_registration_payload",
]
