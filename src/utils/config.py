"""
Configuration loader.

Loads YAML config files and merges them with environment variables from .env.
All pipeline scripts get their parameters from here — never from hardcoded
values in the script itself.
"""

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


def load_config(config_path: str | Path) -> dict[str, Any]:
    """
    Load a YAML config file and return as a dictionary.

    Args:
        config_path: Path to the .yaml config file.

    Returns:
        Dictionary of configuration values.

    Raises:
        FileNotFoundError: If the config file does not exist.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config or {}


def load_env(env_file: str = ".env") -> None:
    """
    Load environment variables from a .env file.
    Must be called once at the start of any script that needs env vars.
    """
    env_path = Path(env_file)
    if env_path.exists():
        load_dotenv(env_path)
    else:
        # Try to find .env from project root
        root = _find_project_root()
        if root and (root / ".env").exists():
            load_dotenv(root / ".env")


def get_env(key: str, default: str | None = None) -> str | None:
    """
    Get an environment variable with an optional default.

    Args:
        key:     The environment variable name (e.g. 'CUBICASA_ROOT').
        default: Value to return if key is not set.

    Returns:
        The value as a string, or default.
    """
    return os.environ.get(key, default)


def _find_project_root() -> Path | None:
    """Walk up from cwd to find the project root (contains pyproject.toml)."""
    current = Path.cwd()
    for parent in [current, *current.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    return None