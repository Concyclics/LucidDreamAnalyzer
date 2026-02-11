from __future__ import annotations

import hashlib
from pathlib import Path

from lucid_dream.web_auth import load_allowed_usernames, verify_login



def _hash_for(username: str) -> str:
    return hashlib.sha256((username + "19260817").encode("utf-8")).hexdigest()



def _write_allowed_users(path: Path, users: list[str]) -> None:
    path.write_text(
        '{"allowed_usernames": [' + ",".join(f'"{u}"' for u in users) + "]}",
        encoding="utf-8",
    )



def test_verify_login_accepts_valid_hash(tmp_path: Path) -> None:
    users_path = tmp_path / "allowed_users.json"
    _write_allowed_users(users_path, ["alice"])
    assert verify_login("alice", _hash_for("alice"), allowed_users_path=users_path) is True



def test_verify_login_rejects_invalid_hash(tmp_path: Path) -> None:
    users_path = tmp_path / "allowed_users.json"
    _write_allowed_users(users_path, ["alice"])
    assert verify_login("alice", "deadbeef", allowed_users_path=users_path) is False



def test_verify_login_case_insensitive_hash(tmp_path: Path) -> None:
    users_path = tmp_path / "allowed_users.json"
    _write_allowed_users(users_path, ["alice"])
    assert (
        verify_login("alice", _hash_for("alice").upper(), allowed_users_path=users_path)
        is True
    )



def test_verify_login_rejects_empty_fields(tmp_path: Path) -> None:
    users_path = tmp_path / "allowed_users.json"
    _write_allowed_users(users_path, ["alice"])
    assert verify_login("", _hash_for("alice"), allowed_users_path=users_path) is False
    assert verify_login("alice", "", allowed_users_path=users_path) is False



def test_verify_login_rejects_user_not_in_allowlist(tmp_path: Path) -> None:
    users_path = tmp_path / "allowed_users.json"
    _write_allowed_users(users_path, ["muyang", "chenhan"])
    assert verify_login("alice", _hash_for("alice"), allowed_users_path=users_path) is False



def test_load_allowed_usernames_reads_json(tmp_path: Path) -> None:
    users_path = tmp_path / "allowed_users.json"
    _write_allowed_users(users_path, ["muyang", "chenhan"])
    assert load_allowed_usernames(users_path) == {"muyang", "chenhan"}
