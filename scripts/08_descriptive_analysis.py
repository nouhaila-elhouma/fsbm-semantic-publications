"""Analyses descriptives du dataset collecté → outputs/figures/*.png + outputs/reports/descriptive_summary.json.

Exemple :
    python scripts/08_descriptive_analysis.py
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
from _bootstrap import ROOT  # noqa: F401

from src.analysis.descriptive_analysis import run_descriptive_analysis
from src.utils.config import load_settings, resolve_path
from src.utils.logger import setup_logging


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, help="Fichier settings.yaml alternatif")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    settings = load_settings(args.config)
    log = setup_logging("08_descriptive_analysis", resolve_path(settings, "logs_dir"))
    processed, raw = resolve_path(settings, "processed_dir"), resolve_path(settings, "raw_dir")
    needed = [raw / "chercheurs_fsbm.csv", processed / "chercheurs_clean.csv", processed / "publications_clean.parquet",
              processed / "researcher_publications.csv"]
    missing = [p for p in needed if not p.exists()]
    if missing:
        log.error("Fichiers manquants : %s — lancez scripts/01 à 03.", [p.name for p in missing])
        return 1
    researchers = pd.read_csv(processed / "chercheurs_clean.csv")
    members = pd.read_csv(raw / "chercheurs_fsbm.csv")
    members = members[members["chercheur_id"].isin(researchers["chercheur_id"])]     # mêmes chercheurs que le dataset
    result = run_descriptive_analysis(
        members, researchers, pd.read_parquet(processed / "publications_clean.parquet"),
        pd.read_csv(processed / "researcher_publications.csv"), resolve_path(settings, "figures_dir"),
        resolve_path(settings, "reports_dir"), settings["data"]["institution"])
    print("\nFigures générées :", *result["figures"], sep="\n  ")
    print("\nRappel : ces classements décrivent le dataset collecté, pas la production totale de la FSBM.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
