"""Étape 2 — Nettoyage, déduplication, validation → data/processed/ + rapport qualité.

Entrées  : data/raw/chercheurs_fsbm.csv, scholars_raw.json, publications_raw.json
Sorties  : chercheurs_clean.csv, publications_clean.{csv,parquet}, researcher_publications.csv,
           dataset_final.json (sans embeddings à ce stade), outputs/reports/data_quality_report.{json,md}

Exemple :
    python scripts/03_clean_data.py
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from _bootstrap import ROOT  # noqa: F401

from src.data.extract_fsbm_members import unique_researchers
from src.data.scholar_ids import read_overrides
from src.preprocessing.data_cleaner import (articles_to_dataframe, build_dataset_final, clean_publications,
                                            clean_researchers, compute_quality_report, quality_report_markdown,
                                            validate_researchers)
from src.utils.config import load_settings, resolve_path
from src.utils.io import atomic_write_text, read_json, write_json
from src.utils.logger import setup_logging

LINK_COLUMNS = ["chercheur_id", "article_id", "raw_pub_id"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--institution", help="Établissement à conserver (défaut : settings.yaml → data.institution)")
    p.add_argument("--config", type=Path, help="Fichier settings.yaml alternatif")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    settings = load_settings(args.config)
    log = setup_logging("03_clean_data", resolve_path(settings, "logs_dir"))
    raw_dir, out_dir = resolve_path(settings, "raw_dir"), resolve_path(settings, "processed_dir")
    reports_dir = resolve_path(settings, "reports_dir")
    out_dir.mkdir(parents=True, exist_ok=True)
    institution = args.institution or settings["data"]["institution"]

    members_csv = raw_dir / "chercheurs_fsbm.csv"
    scholars_raw = read_json(raw_dir / "scholars_raw.json")
    pubs_raw = read_json(raw_dir / "publications_raw.json")
    if not members_csv.exists() or scholars_raw is None or pubs_raw is None:
        log.error("Données brutes manquantes dans %s : lancez d'abord les scripts 01 et 02.", raw_dir)
        return 1

    members = pd.read_csv(members_csv)
    d = settings["data"]
    n_in_pdf = len(unique_researchers(members, institution))
    restrict_to = None
    if d.get("only_with_scholar_id", True):
        # le dataset ne garde que les chercheurs dont le Scholar ID est connu (CSV manuel ou profil déjà collecté)
        manual_ids = {cid: sid for cid, sid in read_overrides(ROOT / settings["scraping"]["overrides_file"]).items() if sid}
        restrict_to = set(manual_ids) | {s["chercheur_id"] for s in scholars_raw if s.get("scholar_id")}
    else:
        manual_ids = {}
    researchers = clean_researchers(members, scholars_raw, institution, restrict_to, manual_ids)
    log.info("%d chercheurs %s dans le PDF ; %d dans le dataset (Scholar ID connu : %s) ; %d profils Scholar appariés",
             n_in_pdf, institution, len(researchers), d.get("only_with_scholar_id", True),
             int((researchers["scholar_profile_status"] == "matched").sum()))

    result = clean_publications(pubs_raw, d["min_abstract_chars"], d["year_min"], d["fuzzy_title_threshold"],
                                valid_researchers=set(researchers["chercheur_id"]))
    result["stats"].update(researchers_in_pdf=n_in_pdf, researchers_with_scholar_id=len(researchers))
    articles_df = articles_to_dataframe(result["articles"])
    links_df = pd.DataFrame(result["links"], columns=LINK_COLUMNS)
    log.info("%s", result["stats"])

    validation_errors = validate_researchers(researchers) + result["validation_errors"]
    pd.DataFrame(validation_errors, columns=["article_id", "chercheur_id", "field", "message"]).to_csv(
        reports_dir / "validation_errors.csv", index=False)
    if validation_errors:
        log.warning("%d erreurs de validation (voir validation_errors.csv)", len(validation_errors))

    # --- exports (les listes sont sérialisées en JSON dans les CSV, conservées telles quelles dans le Parquet)
    researchers_csv = researchers.copy()
    researchers_csv["interests"] = researchers_csv["interests"].map(lambda v: "; ".join(v))
    researchers_csv.to_csv(out_dir / "chercheurs_clean.csv", index=False, encoding="utf-8")

    articles_csv = articles_df.copy()
    for col in ("auteurs", "chercheur_ids"):
        articles_csv[col] = articles_csv[col].map(lambda v: json.dumps(list(v), ensure_ascii=False))
    articles_csv.to_csv(out_dir / "publications_clean.csv", index=False, encoding="utf-8")
    articles_df.to_parquet(out_dir / "publications_clean.parquet", index=False)
    links_df.to_csv(out_dir / "researcher_publications.csv", index=False, encoding="utf-8")

    dataset = build_dataset_final(researchers, articles_df, links_df)
    write_json(out_dir / "dataset_final.json", dataset)

    report = compute_quality_report(researchers, articles_df, links_df, pubs_raw, scholars_raw, result["stats"], validation_errors)
    write_json(reports_dir / "data_quality_report.json", report)
    atomic_write_text(reports_dir / "data_quality_report.md", quality_report_markdown(report))

    print("\n=== Rapport qualité ===")
    for key in ("chercheurs_fsbm_dans_pdf", "chercheurs_avec_scholar_id", "chercheurs_fsbm", "chercheurs_par_statut_scholar", "pct_chercheurs_sans_scholar", "publications_uniques",
                "doublons_fusionnes", "publications_avec_doi", "abstracts_recuperes", "pct_publications_sans_abstract",
                "abstracts_par_statut", "erreurs_scraping", "erreurs_validation"):
        print(f"{key:34s}: {report[key]}")
    print(f"\nFichiers écrits dans {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
