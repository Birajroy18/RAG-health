"""Load API keys from secrets.toml into the environment (local development)."""

from __future__ import annotations

import os
from pathlib import Path

_INITIALIZED = False
_SECRET_KEYS = (
    "OPENROUTER_API_KEY",
    "OPENROUTER_MODEL",
    "OPENROUTER_FALLBACK_MODELS",
    "GEMINI_API_KEY",
    "GEMINI_MODEL",
    "NCBI_API_KEY",
    "NCBI_EMAIL",
    "NCBI_TOOL",
)

_PROJECT_DIR = Path(__file__).resolve().parent
_SECRET_PATHS = (
    _PROJECT_DIR / "secrets.toml",
    _PROJECT_DIR / ".streamlit" / "secrets.toml",
)


def _read_toml(path: Path) -> dict[str, str]:
    try:
        import tomllib
    except ImportError:
        raise RuntimeError("Python 3.11+ is required to read secrets.toml (tomllib).") from None

    with path.open("rb") as f:
        data = tomllib.load(f)

    secrets: dict[str, str] = {}
    for key in _SECRET_KEYS:
        value = data.get(key)
        if value is not None and str(value).strip():
            secrets[key] = str(value).strip()
    return secrets


def init_secrets() -> None:
    """Load secrets.toml once; never override variables already set in the environment."""
    global _INITIALIZED
    if _INITIALIZED:
        return
    _INITIALIZED = True

    for path in _SECRET_PATHS:
        if not path.is_file():
            continue
        for key, value in _read_toml(path).items():
            os.environ.setdefault(key, value)
        return


def get_secret(key: str) -> str:
    """Return a secret from the environment (after loading secrets.toml)."""
    init_secrets()
    return os.getenv(key, "").strip()
