"""Étape 3c — Démonstration de recherche sémantique (publications ET chercheurs) avec zembed-1.

Sans --query, les 5 requêtes de démonstration du sujet sont exécutées. Les résultats sont aussi écrits
dans outputs/reports/demo_search_results.{md,json}.

Exemples :
    python scripts/06_demo_search.py
    python scripts/06_demo_search.py --query "natural language processing" --top-k 5 --researchers
    python scripts/06_demo_search.py --interactive
"""
import argparse
import sys
from pathlib import Path

from _bootstrap import ROOT  # noqa: F401

from src.embeddings.zembed_client import EmbeddingError, MissingAPIKeyError
from src.search.demo_queries import DEMO_QUERIES
from src.search.semantic_search import get_default_engine
from src.utils.config import load_settings, resolve_path
from src.utils.io import write_json
from src.utils.logger import setup_logging



def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--query", "-q", action="append", help="Requête (répétable)")
    p.add_argument("--top-k", type=int, help="Nombre de résultats (défaut : settings.yaml → search.top_k)")
    p.add_argument("--researchers", action="store_true", help="Afficher aussi le classement des chercheurs")
    p.add_argument("--interactive", action="store_true", help="Boucle de requêtes interactive")
    p.add_argument("--config", type=Path, help="Fichier settings.yaml alternatif")
    return p.parse_args()


def format_publication(r: dict) -> str:
    who = ", ".join(r["chercheurs"]) or "—"
    labs = " | ".join(r["laboratoires"]) or "—"
    lines = [f"  {r['rank']}. [{r['score']:.3f}] {r['titre']}",
             f"     Chercheurs FSBM : {who}", f"     Labo/équipe : {labs} / {' | '.join(r['equipes']) or '—'}",
             f"     {r['annee'] or 's.d.'} · {r['journal'] or 'revue inconnue'} · {r['citations'] if r['citations'] is not None else '?'} citations"
             f" · source embedding : {r['embedding_source']}"]
    if r["abstract_court"]:
        lines.append(f"     Abstract : {r['abstract_court']}")
    if r["url"]:
        lines.append(f"     {r['url']}")
    return "\n".join(lines)


def run_query(engine, query: str, top_k: int, with_researchers: bool, settings: dict) -> dict:
    pubs = [r.to_dict() for r in engine.search(query, top_k)]
    out = {"query": query, "publications": pubs}
    print(f"\n━━ « {query} » ━━\nPublications :")
    print("\n".join(format_publication(r) for r in pubs))
    if with_researchers:
        res = [r.to_dict() for r in engine.search_researchers(query, top_k, settings["search"]["researcher_pool"],
                                                              settings["search"]["researcher_top_m"])]
        out["chercheurs"] = res
        print("Chercheurs (agrégation des meilleures publications) :")
        for r in res:
            print(f"  {r['rank']}. [{r['score']:.3f}] {r['nom_complet']} — {r['laboratoire'] or 'labo ?'} "
                  f"({r['n_matching_publications']} publications proches)")
    return out


def markdown_report(results: list[dict]) -> str:
    lines = ["# Démonstration de recherche sémantique (zembed-1)", ""]
    for res in results:
        lines += [f"## « {res['query']} »", "", "| # | Score | Titre | Chercheurs FSBM | Année |", "|---|---|---|---|---|"]
        for r in res["publications"]:
            lines.append(f"| {r['rank']} | {r['score']:.3f} | {r['titre'].replace('|', '/')} | {', '.join(r['chercheurs'])} | {r['annee'] or ''} |")
        if res.get("chercheurs"):
            lines += ["", "| # | Score chercheur | Chercheur | Laboratoire | Publications proches |", "|---|---|---|---|---|"]
            lines += [f"| {r['rank']} | {r['score']:.3f} | {r['nom_complet']} | {r['laboratoire'] or ''} | {r['n_matching_publications']} |"
                      for r in res["chercheurs"]]
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    settings = load_settings(args.config)
    log = setup_logging("06_demo_search", resolve_path(settings, "logs_dir"))
    top_k = args.top_k or settings["search"]["top_k"]
    try:
        engine = get_default_engine()
    except MissingAPIKeyError as exc:
        log.error("%s", exc)
        return 1
    except FileNotFoundError as exc:
        log.error("Index ou données manquants (%s) : lancez les scripts 03, 04 et 05.", exc)
        return 1

    queries = args.query or ([] if args.interactive else DEMO_QUERIES)
    results: list[dict] = []
    try:
        for query in queries:
            results.append(run_query(engine, query, top_k, args.researchers or not args.query, settings))
        while args.interactive:
            query = input("\nRequête (vide pour quitter) > ").strip()
            if not query:
                break
            results.append(run_query(engine, query, top_k, True, settings))
    except EmbeddingError as exc:
        log.error("%s", exc)
        return 1
    if results:
        reports = resolve_path(settings, "reports_dir")
        write_json(reports / "demo_search_results.json", results)
        (reports / "demo_search_results.md").write_text(markdown_report(results), encoding="utf-8")
        print(f"\nRésultats sauvegardés dans {reports}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
