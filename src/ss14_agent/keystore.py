"""Saving API keys pasted into the control window.

On macOS keys go in the login Keychain (service "ss14-agent", account = the
key's variable name), so they're encrypted and never written into the repo.
Elsewhere, or if the Keychain refuses, they go in the git-ignored .env file
(readable only by you). The `security` tool takes the key as an argument, so it
is briefly visible to other processes of your own user while it saves.
At startup `load_saved_keys` puts saved keys into the environment, where the
clients read them. A key already set in the environment always wins.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path

from .config import REPO_ROOT

log = logging.getLogger(__name__)
SERVICE = "ss14-agent"


def _keychain() -> bool:
    return sys.platform == "darwin"


def _env_path() -> Path:
    return REPO_ROOT / ".env"


def _write_env(var: str, value: str) -> None:
    path = _env_path()
    lines = path.read_text().splitlines() if path.exists() else []
    lines = [ln for ln in lines if not ln.strip().startswith(f"{var}=")]
    if value:
        lines.append(f"{var}={value}")
    path.write_text("\n".join(lines) + ("\n" if lines else ""))
    try:
        path.chmod(0o600)
    except OSError:
        pass


def save_key(var: str, value: str) -> str:
    """Store a key and put it in the environment now. Returns where it was saved."""
    value = value.strip()
    if value:
        os.environ[var] = value
    else:
        os.environ.pop(var, None)
    if _keychain():
        try:
            if value:
                subprocess.run(["security", "add-generic-password", "-U", "-s", SERVICE, "-a", var, "-w", value],
                               check=True, capture_output=True, timeout=10)
            else:
                subprocess.run(["security", "delete-generic-password", "-s", SERVICE, "-a", var],
                               check=False, capture_output=True, timeout=10)
            return "Keychain"
        except (OSError, subprocess.SubprocessError) as exc:
            log.warning("Keychain save failed (%s); using .env instead", exc)
    _write_env(var, value)
    return ".env"


def _read_keychain(var: str) -> str:
    try:
        out = subprocess.run(["security", "find-generic-password", "-s", SERVICE, "-a", var, "-w"],
                             check=False, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return ""
    return out.stdout.strip() if out.returncode == 0 else ""


def load_saved_keys(variables: list[str]) -> None:
    """Fill os.environ from the Keychain for any of these variables not already set."""
    if not _keychain():
        return  # .env is loaded by config.load_dotenv
    for var in variables:
        if var and not os.environ.get(var):
            value = _read_keychain(var)
            if value:
                os.environ[var] = value
