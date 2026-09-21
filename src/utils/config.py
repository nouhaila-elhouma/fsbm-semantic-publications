"""Chargement de la configuration (settings.yaml + variables d'environnement)."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SETTINGS_PATH = PROJECT_ROOT / "config" / "settings.yaml"


def load_settings(path: Path | str | None = None) -> dict[str, Any]:
    """Charge ``settings.yaml`` et le fichier ``.env`` (s'il existe)."""
    load_dotenv(PROJECT_ROOT / ".env")
    settings_path = Path(path) if path else DEFAULT_SETTINGS_PATH
    with settings_path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def resolve_path(settings: dict[str, Any], key: str) -> Path:
    """Retourne le chemin absolu associé à ``settings['paths'][key]``."""
    return (PROJECT_ROOT / settings["paths"][key]).resolve()


def get_env(name: str, default: str | None = None) -> str | None:
    """Lit une variable d'environnement en traitant la chaîne vide comme absente."""
    value = os.getenv(name, default)
    return value if value else default
