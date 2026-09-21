"""Prépare l'archive à envoyer sur Google Colab pour encoder les publications avec zembed-1 sur GPU.

Contenu : src/, scripts/, config/settings.yaml, requirements.txt et UNIQUEMENT data/processed/publications_clean.parquet
(titres, abstracts, identifiants). Ni .env, ni PDF, ni données brutes, ni cache Scholar.

Exemple :
    python scripts/make_colab_bundle.py          # écrit outputs/colab/colab_bundle.zip
"""
import argparse
import sys
import zipfile
from pathlib import Path

from _bootstrap import ROOT

from src.utils.config import load_settings, resolve_path

INCLUDE_DIRS = ["src", "scripts", "config"]
INCLUDE_FILES = ["requirements.txt", "pytest.ini"]
SKIP_PARTS = {"__pycache__", ".pytest_cache"}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", type=Path, default=ROOT / "outputs" / "colab" / "colab_bundle.zip")
    args = p.parse_args()
    settings = load_settings()
    parquet = resolve_path(settings, "processed_dir") / "publications_clean.parquet"
    if not parquet.exists():
        print(f"{parquet} introuvable : lancez d'abord scripts/03_clean_data.py")
        return 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, "w", zipfile.ZIP_DEFLATED) as zf:
        for folder in INCLUDE_DIRS:
            for path in sorted((ROOT / folder).rglob("*")):
                if path.is_file() and not (SKIP_PARTS & set(path.parts)) and path.suffix != ".pyc":
                    zf.write(path, path.relative_to(ROOT).as_posix())
        for name in INCLUDE_FILES:
            if (ROOT / name).exists():
                zf.write(ROOT / name, name)
        zf.write(parquet, parquet.relative_to(ROOT).as_posix())
        zf.writestr("data/vector_store/.gitkeep", "")
    size = args.output.stat().st_size / 1e6
    print(f"✔ {args.output} ({size:.1f} Mo) — à envoyer dans notebooks/02_colab_embeddings_gpu.ipynb")
    return 0


if __name__ == "__main__":
    sys.exit(main())
