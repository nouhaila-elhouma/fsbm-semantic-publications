"""Importe des pages de profil Google Scholar ENREGISTRÉES À LA MAIN (lecture hors ligne, aucune requête vers Google).

Marche à suivre pour la personne qui collecte :
  1. python scripts/import_scholar_html.py --checklist      → outputs/reports/profils_a_enregistrer.html (liens cliquables)
  2. ouvrir chaque profil dans le navigateur, valider soi-même un éventuel contrôle « not a robot », attendre l'affichage
     des publications, puis Ctrl+S → « Page Web, HTML uniquement » dans data/input/scholar_html/ (nom par défaut) ;
  3. python scripts/import_scholar_html.py               → lit les fichiers, met à jour les données, enrichit les abstracts.

Rythme conseillé : un profil toutes les 20-30 secondes, sans extension ni outil d'automatisation.
Un profil déjà collecté depuis Scholar n'est pas écrasé (sauf --force). Les données Scholar prennent la priorité sur le repli OpenAlex.
"""
import argparse
import html as htmllib
import sys
from pathlib import Path

import pandas as pd
from _bootstrap import ROOT

from src.data.collection_steps import run_enrichment
from src.data.extract_fsbm_members import unique_researchers
from src.data.scholar_ids import read_overrides
from src.data.scholar_saved_pages import inspect_saved_page, read_html, state_from_saved_page
from src.data.scholar_scraper import ScholarCollector, make_backend
from src.utils.config import load_settings, resolve_path
from src.utils.logger import setup_logging


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--folder", type=Path, help="Dossier des pages enregistrées (défaut : data/input/scholar_html)")
    p.add_argument("--max-publications", type=int, default=30, help="Publications max par chercheur (défaut 30)")
    p.add_argument("--force", action="store_true", help="Remplacer aussi les chercheurs déjà collectés depuis Scholar")
    p.add_argument("--checklist", action="store_true", help="Écrit la liste cliquable des profils à enregistrer, puis s'arrête")
    p.add_argument("--skip-enrichment", action="store_true", help="Ne pas compléter DOI/abstracts via Crossref/OpenAlex/Semantic Scholar")
    p.add_argument("--config", type=Path, help="Fichier settings.yaml alternatif")
    return p.parse_args()


def has_scholar_data(state: dict | None) -> bool:
    return bool(state and state.get("scholar_profile_status") == "matched" and state.get("publications")
                and state.get("data_source", "google_scholar") == "google_scholar")


def write_checklist(records: list[dict], ids: dict, collector: ScholarCollector, path: Path) -> int:
    rows, todo = [], 0
    for rec in records:
        sid = ids[rec["chercheur_id"]]
        done = has_scholar_data(collector.states.get(rec["chercheur_id"]))
        todo += not done
        url = f"https://scholar.google.com/citations?user={sid}&hl=en"
        rows.append(f"<tr class='{'ok' if done else 'todo'}'><td>{'✔ fait' if done else 'à faire'}</td>"
                    f"<td>{htmllib.escape(rec['nom_complet'])}</td><td>{htmllib.escape(rec.get('laboratoire') or '')}</td>"
                    f"<td><a href='{url}' target='_blank' rel='noopener'>ouvrir le profil</a></td></tr>")
    page = ("<!doctype html><meta charset='utf-8'><title>Profils Scholar à enregistrer</title><style>"
            "body{font:15px system-ui;margin:2rem auto;max-width:60rem;padding:0 1rem}td,th{padding:.35rem .6rem;border-bottom:1px solid #ddd;text-align:left}"
            "tr.ok{color:#888}a{color:#1a56b0}</style>"
            f"<h1>Profils Google Scholar à enregistrer ({todo} à faire sur {len(records)})</h1>"
            "<ol><li>Cliquez sur « ouvrir le profil » (un profil toutes les 20-30 s environ).</li>"
            "<li>Si Google affiche « not a robot », validez le contrôle vous-même, puis attendez l'affichage des publications.</li>"
            "<li><b>Ctrl+S</b> → type « Page Web, HTML uniquement » → dossier <code>data/input/scholar_html/</code> (nom par défaut).</li>"
            "<li>Ensuite : <code>python scripts/import_scholar_html.py</code></li></ol>"
            "<table><tr><th>État</th><th>Chercheur</th><th>Laboratoire</th><th>Lien</th></tr>" + "".join(rows) + "</table>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(page, encoding="utf-8")
    return todo


def main() -> int:
    args = parse_args()
    settings = load_settings(args.config)
    log = setup_logging("import_scholar_html", resolve_path(settings, "logs_dir"))
    members_csv = resolve_path(settings, "raw_dir") / "chercheurs_fsbm.csv"
    if not members_csv.exists():
        log.error("%s introuvable : lancez d'abord scripts/01_extract_members.py", members_csv)
        return 1
    records = unique_researchers(pd.read_csv(members_csv), settings["data"]["institution"])
    ids = {cid: sid for cid, sid in read_overrides(ROOT / settings["scraping"]["overrides_file"]).items() if sid}
    records = [r for r in records if r["chercheur_id"] in ids]
    by_scholar_id = {ids[r["chercheur_id"]]: r for r in records}
    collector = ScholarCollector(make_backend(settings), settings, resolve_path(settings, "raw_dir"), resolve_path(settings, "cache_dir"),
                                 roster_names=[r["nom_complet"] for r in records], overrides_path=ROOT / settings["scraping"]["overrides_file"])

    if args.checklist:
        todo = write_checklist(records, ids, collector, resolve_path(settings, "reports_dir") / "profils_a_enregistrer.html")
        print(f"Liste écrite : outputs/reports/profils_a_enregistrer.html ({todo} profils à enregistrer sur {len(records)}).\n"
              "Ouvrez ce fichier dans votre navigateur et suivez les consignes affichées en haut de la page.")
        return 0

    folder = args.folder or ROOT / "data" / "input" / "scholar_html"
    files = sorted(folder.glob("*.htm*")) if folder.exists() else []
    if not files:
        print(f"Aucun fichier .html dans {folder}. Lancez d'abord : python scripts/import_scholar_html.py --checklist")
        return 1

    report = {"importes": [], "deja_scholar": [], "controle_robot": [], "pas_un_profil": [], "id_inconnu": []}
    for file in files:
        info = inspect_saved_page(read_html(file))
        if info["status"] == "robot_check":
            report["controle_robot"].append(file.name)
            continue
        if info["status"] == "not_a_profile":
            report["pas_un_profil"].append(file.name)
            continue
        rec = by_scholar_id.get(info["scholar_id"])
        if rec is None:
            report["id_inconnu"].append(f"{file.name} (ID {info['scholar_id']})")
            continue
        cid = rec["chercheur_id"]
        if has_scholar_data(collector.states.get(cid)) and not args.force:
            report["deja_scholar"].append(rec["nom_complet"])
            continue
        state = state_from_saved_page(collector._new_state(rec), info["scholar_id"], info["profile"], args.max_publications, file.name)
        collector.save_state(state)
        report["importes"].append(f"{rec['nom_complet']} ({len(state['publications'])} publications)")
    collector.write_consolidated()

    print(f"\n=== Import des pages enregistrées ({len(files)} fichiers) ===")
    print(f"Importés : {len(report['importes'])}")
    for line in report["importes"]:
        print("   +", line)
    for key, title in (("deja_scholar", "Déjà collectés depuis Scholar (non écrasés, --force pour remplacer)"),
                       ("controle_robot", "Pages « not a robot » (à refaire : validez le contrôle puis réenregistrez)"),
                       ("pas_un_profil", "Fichiers qui ne sont pas un profil Scholar"), ("id_inconnu", "Profil hors de la liste")):
        if report[key]:
            print(f"{title} : {len(report[key])}")
            for line in report[key][:15]:
                print("   -", line)
    if report["importes"] and not args.skip_enrichment and settings["enrichment"]["enabled"]:
        collector = ScholarCollector(make_backend(settings), settings, resolve_path(settings, "raw_dir"), resolve_path(settings, "cache_dir"),
                                     overrides_path=ROOT / settings["scraping"]["overrides_file"])
        run_enrichment(collector, settings, log)
    remaining = sum(1 for r in records if not has_scholar_data(collector.states.get(r["chercheur_id"])))
    print(f"\nChercheurs avec des données Google Scholar : {len(records) - remaining}/{len(records)} — il reste {remaining} profils à enregistrer.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
