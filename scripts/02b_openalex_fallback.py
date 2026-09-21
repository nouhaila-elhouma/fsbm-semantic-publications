"""Étape 1 bis — Repli OpenAlex pour les chercheurs que Google Scholar n'a pas pu fournir (HTTP 429, blocage).

Pourquoi : le projet ne contourne JAMAIS les limitations de Scholar (pas de proxy, pas de changement d'IP). Pour ne pas
rester bloqué, les chercheurs sans données Scholar sont complétés via l'API officielle et ouverte d'OpenAlex
(publications, abstracts complets, DOI, revues, citations, h-index). Ces données sont étiquetées
``data_source = "openalex"`` ; dès que Scholar fournit des publications pour un chercheur, Scholar reprend la priorité.

Le profil OpenAlex n'est retenu que si son nom est quasi identique ET s'il est rattaché à l'Université Hassan II de
Casablanca ; les homonymes importants sont écartés (« ambiguous »), les fragments d'un même auteur sont fusionnés.

Exemples :
    python scripts/02b_openalex_fallback.py --validate     # compare OpenAlex à Scholar sur les chercheurs déjà collectés
    python scripts/02b_openalex_fallback.py                # repli pour tous les chercheurs sans données Scholar
    python scripts/02b_openalex_fallback.py --only "Benlahmar" --refresh
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
from _bootstrap import ROOT

from src.data.collection_steps import run_enrichment
from src.data.extract_fsbm_members import unique_researchers
from src.data.openalex_source import OpenAlexFallback, OpenAlexSource
from src.data.scholar_ids import read_overrides
from src.data.scholar_scraper import ScholarCollector, make_backend
from src.preprocessing.text_cleaner import normalize_for_matching, title_similarity
from src.utils.config import get_env, load_settings, resolve_path
from src.utils.io import read_json, write_json
from src.utils.logger import setup_logging


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--max-publications", type=int, default=30, help="Publications max par chercheur, triées par citations (défaut 30)")
    p.add_argument("--only", action="append", metavar="TEXTE", help="Ne traiter que les chercheurs dont le nom/l'id contient TEXTE")
    p.add_argument("--limit-researchers", type=int, metavar="N", help="Mode test : les N premiers chercheurs")
    p.add_argument("--refresh", action="store_true", help="Ignorer le cache OpenAlex existant")
    p.add_argument("--validate", action="store_true", help="Comparer OpenAlex à Scholar sur les chercheurs déjà collectés (rien n'est écrit dans les données)")
    p.add_argument("--skip-enrichment", action="store_true", help="Ne pas compléter les abstracts manquants via Crossref/S2")
    p.add_argument("--config", type=Path, help="Fichier settings.yaml alternatif")
    return p.parse_args()


def scholar_has_data(state: dict | None) -> bool:
    return bool(state and state.get("scholar_profile_status") == "matched" and state.get("publications"))


def validate(fallback: OpenAlexFallback, collector: ScholarCollector, records: list[dict], alt_names: dict, ids: dict,
             max_pubs: int, log) -> dict:
    """Recouvrement OpenAlex ↔ Scholar : quelle part des publications Scholar retrouve-t-on dans OpenAlex ?"""
    rows = []
    for rec in records:
        state = collector.states.get(rec["chercheur_id"])
        if not scholar_has_data(state):
            continue
        oa = fallback.source.collect(rec, [rec["nom_complet"], *alt_names.get(rec["chercheur_id"], [])], ids.get(rec["chercheur_id"]), max_pubs)
        if oa["scholar_profile_status"] != "matched":
            rows.append({"chercheur": rec["nom_complet"], "openalex": oa["scholar_profile_status"], "detail": oa["status_detail"]})
            continue
        oa_dois = {p["doi"] for p in oa["publications"] if p.get("doi")}
        oa_titles = [p["titre"] for p in oa["publications"]]
        found = 0
        for sp in state["publications"]:
            same_doi = bool(sp.get("doi")) and sp["doi"] in oa_dois
            found += same_doi or any(title_similarity(sp["titre"], t) >= 0.9 for t in oa_titles)
        sm, om = state["profile"]["metrics"], oa["profile"]["metrics"]
        rows.append({"chercheur": rec["nom_complet"], "openalex": "matched", "scholar_publications": len(state["publications"]),
                     "retrouvees_dans_openalex": int(found), "recall": round(found / len(state["publications"]), 2),
                     "openalex_publications": len(oa["publications"]), "openalex_ids_fusionnes": len(oa["openalex_ids"]),
                     "h_index_scholar": sm.get("h_index"), "h_index_openalex": om.get("h_index"),
                     "citations_scholar": sm.get("citations"), "citations_openalex": om.get("citations")})
    matched = [r for r in rows if r.get("recall") is not None]
    summary = {"chercheurs_compares": len(rows), "profils_openalex_trouves": len(matched),
               "recall_moyen": round(sum(r["recall"] for r in matched) / len(matched), 2) if matched else None, "detail": rows}
    log.info("Validation : %s", {k: v for k, v in summary.items() if k != "detail"})
    return summary


def main() -> int:
    args = parse_args()
    settings = load_settings(args.config)
    log = setup_logging("02b_openalex_fallback", resolve_path(settings, "logs_dir"))
    members_csv = resolve_path(settings, "raw_dir") / "chercheurs_fsbm.csv"
    if not members_csv.exists():
        log.error("%s introuvable : lancez d'abord scripts/01_extract_members.py", members_csv)
        return 1
    records = unique_researchers(pd.read_csv(members_csv), settings["data"]["institution"])
    ids = {cid: sid for cid, sid in read_overrides(ROOT / settings["scraping"]["overrides_file"]).items() if sid}
    records = [r for r in records if r["chercheur_id"] in ids]           # périmètre du dataset : chercheurs avec Scholar ID
    if args.only:
        squash = lambda t: normalize_for_matching(t).replace(" ", "")  # noqa: E731
        records = [r for r in records if any(squash(o) in squash(r["nom_complet"] + r["chercheur_id"]) for o in args.only)]
    if args.limit_researchers:
        records = records[: args.limit_researchers]

    # noms « naturels » saisis à la main (target_researchers.json) : meilleures requêtes que les noms collés du PDF
    report = read_json(resolve_path(settings, "reports_dir") / "scholar_ids_import_report.json", default=[])
    alt_names: dict[str, list[str]] = {}
    for item in report:
        if item.get("status") == "accepted":
            alt_names.setdefault(item["chercheur_id"], []).append(item["input_name"])

    collector = ScholarCollector(make_backend(settings), settings, resolve_path(settings, "raw_dir"),
                                 resolve_path(settings, "cache_dir"), roster_names=[r["nom_complet"] for r in records],
                                 overrides_path=ROOT / settings["scraping"]["overrides_file"])
    source = OpenAlexSource(contact_email=get_env("CONTACT_EMAIL"), api_key=get_env("OPENALEX_API_KEY"))
    fallback = OpenAlexFallback(source, resolve_path(settings, "cache_dir"))
    if not get_env("CONTACT_EMAIL"):
        log.warning("CONTACT_EMAIL absent : le « polite pool » d'OpenAlex est recommandé (.env).")

    if args.validate:
        summary = validate(fallback, collector, records, alt_names, ids, max(args.max_publications, 50), log)
        write_json(resolve_path(settings, "reports_dir") / "openalex_vs_scholar_validation.json", summary)
        print("\n=== Validation OpenAlex ↔ Google Scholar ===")
        for r in summary["detail"]:
            print(r)
        print(f"\nRecall moyen (part des publications Scholar retrouvées dans OpenAlex) : {summary['recall_moyen']}")
        return 0

    todo = [r for r in records if not scholar_has_data(collector.states.get(r["chercheur_id"]))]
    log.info("%d chercheurs avec Scholar ID ; %d sans données Scholar → repli OpenAlex", len(records), len(todo))
    counts = fallback.run(todo, alt_names, ids, args.max_publications, resume=not args.refresh)
    collector.fallback_states = fallback.states                  # les états viennent d'être (re)calculés
    if not args.skip_enrichment and settings["enrichment"]["enabled"]:
        run_enrichment(collector, settings, log)
    else:
        collector.write_consolidated()

    eff = collector.effective_states()
    by_source: dict[str, int] = {}
    for r in records:
        st = eff.get(r["chercheur_id"], {})
        key = st.get("data_source", "google_scholar") if st.get("scholar_profile_status") == "matched" else "sans_donnees"
        by_source[key] = by_source.get(key, 0) + 1
    n_pubs = sum(len(eff[r["chercheur_id"]].get("publications", [])) for r in records if r["chercheur_id"] in eff)
    print("\n=== Repli OpenAlex ===")
    print(f"Statuts OpenAlex : {counts}")
    print(f"Chercheurs par source de données : {by_source} | publications brutes : {n_pubs}")
    missing = [r["nom_complet"] for r in todo if fallback.states.get(r["chercheur_id"], {}).get("scholar_profile_status") != "matched"]
    if missing:
        print(f"Sans données (profil OpenAlex introuvable/ambigu) : {len(missing)} → {', '.join(missing[:12])}{'…' if len(missing) > 12 else ''}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
