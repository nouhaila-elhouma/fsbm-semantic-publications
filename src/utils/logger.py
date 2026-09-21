"""Configuration du logging : console + fichier dans ``outputs/logs/``."""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

LOG_FORMAT = "%(asctime)s %(levelname)-7s %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M"


def setup_logging(name: str, log_dir: Path | str, level: int = logging.INFO) -> logging.Logger:
    """Crée (ou reconfigure) le logger racine du projet.

    Un fichier ``<name>_<YYYYMMDD>.log`` est ajouté dans ``log_dir`` ; les messages
    sont aussi affichés dans la console. Appel idempotent.
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.setLevel(level)
    for handler in list(root.handlers):
        root.removeHandler(handler)

    formatter = logging.Formatter(LOG_FORMAT, DATE_FORMAT)
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    root.addHandler(console)

    log_file = log_dir / f"{name}_{datetime.now():%Y%m%d}.log"
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    # Bibliothèques bavardes
    for noisy in ("urllib3", "httpx", "httpcore", "matplotlib", "numba", "PIL", "pdfminer", "pdfplumber"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    return logging.getLogger(name)


def get_logger(name: str) -> logging.Logger:
    """Raccourci pour obtenir un logger nommé (à utiliser dans les modules)."""
    return logging.getLogger(name)
