"""Authentication helpers for the Streamlit web app."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
from pathlib import Path


LOGIN_SALT = "19260817"
DEFAULT_ALLOWED_USERS_PATH = Path("config/allowed_users.json")



def load_allowed_usernames(path: Path | None = None) -> set[str]:
    """Load allowed usernames from JSON file."""

    users_path = path or Path(
        os.getenv("LUCID_ALLOWED_USERS_JSON", str(DEFAULT_ALLOWED_USERS_PATH))
    )
    if not users_path.exists():
        return set()

    with users_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    usernames = payload.get("allowed_usernames")
    if not isinstance(usernames, list):
        raise ValueError(f"Invalid allowed users JSON: {users_path}")
    return {str(item).strip() for item in usernames if str(item).strip()}



def verify_login(
    username: str,
    provided_hash: str,
    allowed_users_path: Path | None = None,
) -> bool:
    """Verify username and SHA256 hash credentials.

    A login is valid when:
    sha256((username + "19260817").encode()).hexdigest() == provided_hash

    Comparison is case-insensitive and constant-time.
    """

    user = (username or "").strip()
    supplied = (provided_hash or "").strip()
    if not user or not supplied:
        return False
    if user not in load_allowed_usernames(allowed_users_path):
        return False

    expected = hashlib.sha256((user + LOGIN_SALT).encode("utf-8")).hexdigest()
    return hmac.compare_digest(expected, supplied.lower())
