"""Officer / administrator accounts: scrypt password hashing, lockout after repeated failures."""

from __future__ import annotations

import base64
import hashlib
import hmac
import re
import secrets
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional, Tuple

from storage import Storage

ROLES = ("officer", "admin")
USERNAME_RE = re.compile(r"^[A-Za-z0-9._-]{3,40}$")
MIN_PASSWORD_LEN = 8
MAX_FAILED_ATTEMPTS = 5
LOCKOUT_MINUTES = 15

_SCRYPT = {"n": 2 ** 14, "r": 8, "p": 1, "dklen": 32}
_DUMMY_HASH: Optional[str] = None


def _now() -> datetime:
    return datetime.now(timezone.utc)


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    dk = hashlib.scrypt(password.encode("utf-8"), salt=salt, **_SCRYPT)
    return "scrypt$" + base64.b64encode(salt).decode() + "$" + base64.b64encode(dk).decode()


def verify_password(password: str, stored: str) -> bool:
    try:
        scheme, salt_b64, dk_b64 = stored.split("$")
        if scheme != "scrypt":
            return False
        dk = hashlib.scrypt(password.encode("utf-8"), salt=base64.b64decode(salt_b64), **_SCRYPT)
        return hmac.compare_digest(dk, base64.b64decode(dk_b64))
    except (ValueError, TypeError):
        return False


def validate_new_user(username: str, full_name: str, password: str, role: str) -> Optional[str]:
    if not USERNAME_RE.match(username or ""):
        return "Username must be 3-40 characters: letters, digits, '.', '_' or '-'."
    if not (full_name or "").strip():
        return "Full name is required."
    if len(password or "") < MIN_PASSWORD_LEN:
        return f"Password must be at least {MIN_PASSWORD_LEN} characters."
    if role not in ROLES:
        return "Invalid role."
    return None


class AuthService:
    def __init__(self, storage: Storage) -> None:
        self.storage = storage

    def has_users(self) -> bool:
        return self.storage.count_users() > 0

    def create_user(self, username: str, full_name: str, password: str, role: str = "officer") -> Optional[str]:
        username = (username or "").strip()
        err = validate_new_user(username, full_name, password, role)
        if err:
            return err
        if self.storage.get_user(username):
            return f"User '{username}' already exists."
        self.storage.create_user({
            "username": username, "full_name": full_name.strip(), "role": role,
            "password_hash": hash_password(password), "active": True, "failed_attempts": 0,
            "locked_until": None, "created_at": _now().isoformat(timespec="seconds"),
        })
        return None

    def ensure_bootstrap_admin(self, username: Optional[str], password: Optional[str]) -> Optional[str]:
        if not username or not password or self.has_users():
            return None
        return self.create_user(username, "System Administrator", password, "admin")

    def authenticate(self, username: str, password: str) -> Tuple[Optional[Dict[str, Any]], str]:
        global _DUMMY_HASH
        username = (username or "").strip()
        user = self.storage.get_user(username) if USERNAME_RE.match(username) else None
        if user is None:
            if _DUMMY_HASH is None:
                _DUMMY_HASH = hash_password(secrets.token_hex(8))
            verify_password(password or "", _DUMMY_HASH)  # equalise timing for unknown users
            return None, "Invalid username or password."
        if not user["active"]:
            return None, "This account has been deactivated. Contact the administrator."
        if user["locked_until"] and datetime.fromisoformat(user["locked_until"]) > _now():
            return None, f"Account temporarily locked after repeated failed logins. Try again after {LOCKOUT_MINUTES} minutes."
        if not verify_password(password or "", user["password_hash"]):
            attempts = int(user["failed_attempts"]) + 1
            fields: Dict[str, Any] = {"failed_attempts": attempts}
            if attempts >= MAX_FAILED_ATTEMPTS:
                fields = {"failed_attempts": 0,
                          "locked_until": (_now() + timedelta(minutes=LOCKOUT_MINUTES)).isoformat(timespec="seconds")}
            self.storage.update_user(username, **fields)
            return None, "Invalid username or password."
        if user["failed_attempts"] or user["locked_until"]:
            self.storage.update_user(username, failed_attempts=0, locked_until=None)
        return {k: user[k] for k in ("username", "full_name", "role")}, ""

    def current(self, username: str) -> Optional[Dict[str, Any]]:
        """Re-validate a session user on every page load (deactivation takes effect immediately)."""
        user = self.storage.get_user(username)
        if not user or not user["active"]:
            return None
        return {k: user[k] for k in ("username", "full_name", "role")}

    def set_password(self, username: str, password: str) -> Optional[str]:
        if len(password or "") < MIN_PASSWORD_LEN:
            return f"Password must be at least {MIN_PASSWORD_LEN} characters."
        if not self.storage.get_user(username):
            return "User not found."
        self.storage.update_user(username, password_hash=hash_password(password), failed_attempts=0, locked_until=None)
        return None

    def set_active(self, username: str, active: bool) -> None:
        self.storage.update_user(username, active=active, failed_attempts=0, locked_until=None)
