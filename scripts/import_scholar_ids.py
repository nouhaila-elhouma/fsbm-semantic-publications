"""Importe des Scholar ID fournis dans un fichier JSON [{"nom_complet": ..., "chercheur_id": <Scholar ID>}, ...]
vers data/raw/scholar_overrides.csv, après rapprochement prudent avec la liste du PDF.

Rien n'est deviné : les entrées douteuses (ID mal formé, nom absent du PDF, homonyme ambigu, autre
établissement, conflit) sont signalées et NON importées. Les correspondances de nom imparfaites
(score < 0,95) sont importées mais listées « à vérifier ».

Exemples :
    python scripts/import_scholar_ids.py --dry-run          # simulation : affiche seulement le rapport
    python scripts/import_scholar_ids.py                    # importe les entrées acceptées
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
from _bootstrap import ROOT

from src.data.extract_fsbm_members import unique_researchers
from src.data.scholar_ids import match_entries, write_overrides
from src.utils.config import load_settings, resolve_path
from src.utils.io import read_json, write_json
from src.utils.logger import setup_logging


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, default=ROOT / "target_researchers.json", help="Fichier JSON (défaut : target_researchers.json)")
    p.add_argument("--min-score", type=float, default=0.80, help="Score de nom minimal pour accepter un rapprochement (défaut 0,80)")
    p.add_argument("--dry-run", action="store_true", help="Ne rien écrire : afficher uniquement le rapport")
    p.add_argument("--config", type=Path, help="Fichier settings.yaml alternatif")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    settings = load_settings(args.config)
    log = setup_logging("import_scholar_ids", resolve_path(settings, "logs_dir"))
    members_csv = resolve_path(settings, "raw_dir") / "chercheurs_fsbm.csv"
    entries = read_json(args.input)
    if entries is None or not members_csv.exists():
        log.error("Fichier introuvable : %s ou %s (lancez d'abord scripts/01_extract_members.py)", args.input, members_csv)
        return 1

    members = pd.read_csv(members_csv)
    institution = settings["data"]["institution"]
    roster = members.drop_duplicates("chercheur_id")[["chercheur_id", "nom_complet", "etablissement"]].to_dict("records")
    results = match_entries(entries, roster, institution, args.min_score)

    accepted = {r.chercheur_id: r.scholar_id for r in results if r.status == "accepted"}
    repeated = len([r for r in results if r.status == "accepted"]) - len(accepted)
    print(f"\n=== Import des Scholar ID ({len(entries)} entrées) ===")
    print(f"Acceptées (chercheurs FSBM distincts) : {len(accepted)}   [{repeated} entrée(s) répétée(s) avec le même ID]")
    for status, title in (("invalid_id", "ID INVALIDES (non importés)"), ("not_in_roster", "ABSENTS du PDF (non importés)"),
                          ("other_institution", "AUTRES ÉTABLISSEMENTS (hors périmètre FSBM, non importés)"),
                          ("ambiguous", "AMBIGUS (non importés)"), ("conflict", "CONFLITS d'ID (non importés)")):
        rows = [r for r in results if r.status == status]
        if rows:
            print(f"\n{title} : {len(rows)}")
            for r in rows:
                print(f"  - {r.input_name!r} [{r.scholar_id}] → {r.matched_name or '?'} ({r.chercheur_id or '—'}) : {r.note}")
    verify = {(r.input_name, r.scholar_id): r for r in results if r.to_verify}
    if verify:
        print(f"\nÀ VÉRIFIER (importés, mais nom différent du PDF) : {len(verify)}")
        for r in verify.values():
            print(f"  - {r.input_name!r} ≈ {r.matched_name!r} (score {r.score:.2f})")

    report = resolve_path(settings, "reports_dir") / "scholar_ids_import_report.json"
    write_json(report, [r.__dict__ for r in results])
    if args.dry_run:
        print(f"\n(simulation) rapport : {report} — rien n'a été écrit dans le CSV.")
        return 0
    overrides = ROOT / settings["scraping"]["overrides_file"]
    n = write_overrides(unique_researchers(members, institution), overrides, accepted)
    print(f"\n✔ {overrides} mis à jour : {n} chercheur(s) FSBM avec un Scholar ID. Rapport : {report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
