"""Registration webhook payload processing."""

import sqlite3
from typing import Any, Dict, Mapping, Optional

from competition.registration.canonical_id_generator import generate_canonical_team_id
from competition.registration.token_generator import generate_submission_token
from competition.storage import SubmissionStore


def _normalize_payload(payload: Mapping[str, Any]) -> Dict[str, Any]:
    return {str(key).strip().lower(): value for key, value in payload.items()}


def _get_payload_value(normalized_payload: Mapping[str, Any], *candidate_keys: str) -> Optional[str]:
    for key in candidate_keys:
        value = normalized_payload.get(key.strip().lower())
        if value is None:
            continue
        if isinstance(value, str):
            value = value.strip()
        if value == "":
            continue
        return str(value)
    return None


def _agreement_checked(raw_value: Optional[str]) -> bool:
    if raw_value is None:
        return False
    truthy_values = {
        "true",
        "yes",
        "y",
        "1",
        "checked",
        "agree",
        "i agree",
        "agreed",
        "on",
    }
    return str(raw_value).strip().lower() in truthy_values


def process_registration_payload(payload: Mapping[str, Any], store: SubmissionStore) -> Dict[str, Any]:
    """Process registration payload and persist generated team identity/token.

    Payload supports both explicit API keys and your Google Form labels:
    - team_name or Team Name
    - primary_contact_name or Primary contact name
    - primary_contact_email or Primary contact email
    - second_contact_name or Second contact name
    - second_contact_email or Second contact email
    - agreement_to_rules or Agreement to rules
    """
    if not isinstance(payload, Mapping):
        return {
            "status": "error",
            "error_code": "INVALID_PAYLOAD",
            "message": "Payload must be a JSON object.",
        }

    normalized = _normalize_payload(payload)
    team_name = _get_payload_value(normalized, "team_name", "team name")
    primary_contact_name = _get_payload_value(
        normalized, "primary_contact_name", "primary contact name"
    )
    primary_contact_email = _get_payload_value(
        normalized, "primary_contact_email", "primary contact email"
    )
    second_contact_name = _get_payload_value(
        normalized, "second_contact_name", "second contact name"
    )
    second_contact_email = _get_payload_value(
        normalized, "second_contact_email", "second contact email"
    )
    primary_university = _get_payload_value(
        normalized, "primary_university", "primary university"
    )
    secondary_university = _get_payload_value(
        normalized, "secondary_university", "secondary university"
    )
    student_id_1 = _get_payload_value(
        normalized, "student_id_1", "student id 1"
    )
    student_id_2 = _get_payload_value(
        normalized, "student_id_2", "student id 2"
    )
    majors = _get_payload_value(
        normalized, "majors", "major"
    )
    discord_name = _get_payload_value(
        normalized, "discord_name", "discord name", "discord"
    )
    agreement_raw = _get_payload_value(
        normalized,
        "agreement_to_rules",
        "agreement to rules",
    )

    if not team_name:
        return {
            "status": "error",
            "error_code": "MISSING_TEAM_NAME",
            "message": "Team name is required.",
        }
    if not primary_contact_email:
        return {
            "status": "error",
            "error_code": "MISSING_PRIMARY_CONTACT_EMAIL",
            "message": "Primary contact email is required.",
        }
    if not primary_university:
        return {
            "status": "error",
            "error_code": "MISSING_PRIMARY_UNIVERSITY",
            "message": "Primary university is required.",
        }
    if not student_id_1:
        return {
            "status": "error",
            "error_code": "MISSING_STUDENT_ID_1",
            "message": "Student ID for Student 1 is required.",
        }
    if not majors:
        return {
            "status": "error",
            "error_code": "MISSING_MAJORS",
            "message": "Majors information is required.",
        }
    if not _agreement_checked(agreement_raw):
        return {
            "status": "error",
            "error_code": "RULES_NOT_ACCEPTED",
            "message": "Agreement to rules must be accepted.",
        }

    existing_team = store.get_team_by_name(team_name)
    registration_mode = "new"
    if existing_team:
        if existing_team.primary_email.lower() != primary_contact_email.lower():
            return {
                "status": "error",
                "error_code": "TEAM_NAME_ALREADY_REGISTERED",
                "message": "Team name already registered with a different email.",
                "team_name": team_name,
            }
        canonical_team_id = existing_team.canonical_team_id
        registration_mode = "existing"
    else:
        canonical_team_id = generate_canonical_team_id(team_name)

    submission_token = generate_submission_token()

    try:
        store.register_team(
            canonical_team_id=canonical_team_id,
            team_name=team_name,
            primary_email=primary_contact_email,
            token=submission_token,
        )
    except sqlite3.IntegrityError:
        return {
            "status": "error",
            "error_code": "REGISTRATION_CONFLICT",
            "message": "Registration conflict. Please retry or contact organizers.",
            "team_name": team_name,
        }

    return {
        "status": "success",
        "registration_mode": registration_mode,
        "team_name": team_name,
        "primary_contact_name": primary_contact_name,
        "primary_contact_email": primary_contact_email,
        "second_contact_name": second_contact_name,
        "second_contact_email": second_contact_email,
        "primary_university": primary_university,
        "secondary_university": secondary_university,
        "student_id_1": student_id_1,
        "student_id_2": student_id_2,
        "majors": majors,
        "discord_name": discord_name,
        "canonical_team_id": canonical_team_id,
        "submission_token": submission_token,
        "onboarding_email_fields": {
            "team_name": team_name,
            "canonical_team_id": canonical_team_id,
            "submission_token": submission_token,
            "discord_community_link": "<TO_BE_FILLED_BY_ORGANIZER>",
            "submission_constraints_and_format": (
                "Upload exactly one .zip with one agent.py. "
                "No path traversal, no symlinks, no nested archives."
            ),
            "contact_help_channel": "<TO_BE_FILLED_BY_ORGANIZER>",
            "submission_form_link": "<TO_BE_FILLED_BY_ORGANIZER>",
            "custom_email_content": "<TO_BE_FILLED_BY_ORGANIZER>",
        },
    }
