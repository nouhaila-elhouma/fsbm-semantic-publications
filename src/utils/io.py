"""Entrées/sorties robustes : écriture atomique (jamais de fichier à moitié écrit)."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


def _json_default(obj: Any) -> Any:
    """Sérialise numpy / pandas / Path sans perdre les valeurs manquantes."""
    try:
        import numpy as np
        import pandas as pd
    except ImportError:  # pragma: no cover
        raise TypeError(f"Non sérialisable : {type(obj)}")
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if obj is pd.NA or obj is pd.NaT:
        return None
    if isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Non sérialisable : {type(obj)}")


def atomic_write_text(path: Path | str, text: str) -> None:
    """Écrit ``text`` dans un fichier temporaire puis le renomme (opération atomique)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
        os.replace(tmp_name, path)
    except BaseException:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
        raise


def _nan_to_none(obj: Any) -> Any:
    """Remplace récursivement les float NaN/inf par None (JSON valide)."""
    if isinstance(obj, float) and (obj != obj or obj in (float("inf"), float("-inf"))):
        return None
    if isinstance(obj, dict):
        return {k: _nan_to_none(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_nan_to_none(v) for v in obj]
    return obj


def write_json(path: Path | str, data: Any, indent: int | None = 2) -> None:
    """Écrit du JSON UTF-8 valide de façon atomique (NaN → null)."""
    kwargs = dict(ensure_ascii=False, indent=indent, default=_json_default, allow_nan=False)
    try:
        text = json.dumps(data, **kwargs)
    except ValueError:  # NaN / inf présents : on nettoie puis on réessaie
        text = json.dumps(_nan_to_none(data), **kwargs)
    atomic_write_text(path, text)


def read_json(path: Path | str, default: Any = None) -> Any:
    """Lit un fichier JSON ; retourne ``default`` s'il n'existe pas."""
    path = Path(path)
    if not path.exists():
        return default
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)
