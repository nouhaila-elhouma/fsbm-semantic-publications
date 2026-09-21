"""Étape 1 — Collecte Google Scholar (profils + publications) puis enrichissement (DOI, abstracts).

Sorties : data/raw/scholars_raw.json, data/raw/publications_raw.json, data/raw/scholar_candidates_review.csv
Cache/checkpoints : data/raw/cache/ (reprise automatique avec --resume, activé par défaut).

Conformité robots.txt de Google Scholar : les profils (?user=ID) sont autorisés, mais la recherche d'auteurs par
nom et la pagination (cstart=) sont interdites. Par défaut le script n'interroge donc que les profils dont le
Scholar ID est renseigné dans data/raw/scholar_overrides.csv (--init-overrides génère le modèle).

Exemples :
    python scripts/02_scrape_scholar.py --init-overrides                 # modèle des Scholar ID à remplir
    python scripts/02_scrape_scholar.py --limit-researchers 3 --max-publications 10
    python scripts/02_scrape_scholar.py --only "Ben Lahmar" --max-publications 20
    python scripts/02_scrape_scholar.py --resume            # reprise après interruption ou blocage
    python scripts/02_scrape_scholar.py --allow-author-search   # opt-in explicite (interdit par robots.txt)
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
from _bootstrap import ROOT

from src.data.extract_fsbm_members import unique_researchers
from src.data.collection_steps import run_enrichment
from src.data.scholar_ids import write_overrides
from src.data.scholar_scraper import ScholarCollector, make_backend
from src.preprocessing.text_cleaner import normalize_for_matching
from src.utils.config import load_settings, resolve_path
from src.utils.logger import setup_logging


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--limit-researchers", type=int, metavar="N", help="Ne traiter que les N premiers chercheurs (mode test)")
    p.add_argument("--max-publications", type=int, metavar="N", help="Nombre max de publications par chercheur")
    p.add_argument("--only", action="append", metavar="TEXTE", help="Ne traiter que les chercheurs dont le nom/l'id contient TEXTE (répétable)")
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True,
                   help="Reprendre depuis le cache (défaut). --no-resume ignore le cache existant pour les chercheurs traités.")
    p.add_argument("--backend", choices=["http", "scholarly"], help="Méthode de collecte (défaut : settings.yaml)")
    p.add_argument("--no-details", action="store_true", help="Ne pas ouvrir la page de chaque publication (plus rapide, moins d'abstracts)")
    p.add_argument("--skip-enrichment", action="store_true", help="Ne pas interroger Crossref/OpenAlex/Semantic Scholar")
    p.add_argument("--enrich-only", action="store_true", help="Ne faire que l'enrichissement des données déjà collectées")
    p.add_argument("--init-overrides", action="store_true",
                   help="Écrit/rafraîchit le modèle scholar_overrides.csv (un Scholar ID par chercheur, à remplir à la main) puis s'arrête")
    p.add_argument("--allow-author-search", action="store_true",
                   help="ATTENTION : active la recherche automatique d'auteurs par nom, que le robots.txt de Google Scholar interdit "
                        "(Disallow: /citations?). Choix et responsabilité de l'utilisateur ; désactivée par défaut.")
    p.add_argument("--members-csv", type=Path, help="CSV des chercheurs (défaut : data/raw/chercheurs_fsbm.csv)")
    p.add_argument("--config", type=Path, help="Fichier settings.yaml alternatif")
    return p.parse_args()


def select_researchers(records: list[dict], only: list[str] | None, limit: int | None) -> list[dict]:
    if only:
        # insensible aux espaces : « Ben Lahmar » retrouve « Elhabib Benlahmar » (noms collés dans le PDF)
        squash = lambda text: normalize_for_matching(text).replace(" ", "")  # noqa: E731
        needles = [squash(o) for o in only]
        records = [r for r in records if any(n in squash(r["nom_complet"] + " " + r["chercheur_id"]) for n in needles)]
    return records[:limit] if limit else records


def main() -> int:
    args = parse_args()
    settings = load_settings(args.config)
    if args.backend:
        settings["scraping"]["backend"] = args.backend
    if args.no_details:
        settings["scraping"]["fetch_publication_details"] = False
    if args.allow_author_search:
        settings["scraping"]["allow_author_search"] = True
    log = setup_logging("02_scrape_scholar", resolve_path(settings, "logs_dir"))
    max_pubs = args.max_publications or settings["scraping"]["max_publications_per_researcher"]

    members_csv = args.members_csv or resolve_path(settings, "raw_dir") / "chercheurs_fsbm.csv"
    if not members_csv.exists():
        log.error("%s introuvable : lancez d'abord scripts/01_extract_members.py", members_csv)
        return 1
    members = pd.read_csv(members_csv)
    institution = settings["data"]["institution"]
    all_records = unique_researchers(members, institution)
    roster = [r["nom_complet"] for r in all_records]
    overrides_path = ROOT / settings["scraping"]["overrides_file"]
    if args.init_overrides:
        n_filled = write_overrides(all_records, overrides_path)
        print(f"Modèle écrit : {overrides_path} ({len(all_records)} chercheurs, {n_filled} Scholar ID déjà renseignés).\n"
              "Ouvrez le profil Google Scholar de chaque chercheur et copiez la valeur de « user=… » de l'URL "
              "dans la colonne scholar_id (laissez vide si le chercheur n'a pas de profil).")
        return 0

    collector = ScholarCollector(make_backend(settings), settings, resolve_path(settings, "raw_dir"),
                                 resolve_path(settings, "cache_dir"), roster_names=roster, overrides_path=overrides_path)
    pool = all_records
    if not collector.backend.search_allowed:      # sans recherche autorisée, seuls les chercheurs dont l'ID est connu sont traitables
        pool = [r for r in all_records if r["chercheur_id"] in collector.overrides or (collector.states.get(r["chercheur_id"]) or {}).get("scholar_id")]
        log.info("Recherche d'auteurs désactivée (robots.txt) : %d/%d chercheurs ont un Scholar ID renseigné.", len(pool), len(all_records))
        if not pool and not args.enrich_only:
            print("Aucun Scholar ID connu. La recherche automatique par nom est interdite par le robots.txt de Google Scholar.\n"
                  "  1) python scripts/02_scrape_scholar.py --init-overrides   (génère le modèle à remplir)\n"
                  f"  2) renseignez la colonne scholar_id dans {settings['scraping']['overrides_file']}\n"
                  "  ou, sous votre responsabilité, relancez avec --allow-author-search.")
            return 1
    researchers = select_researchers(pool, args.only, args.limit_researchers)
    log.info("%d chercheurs %s dans le CSV ; %d sélectionnés pour cette exécution", len(all_records), institution, len(researchers))
    if not researchers and not args.enrich_only:
        log.error("Aucun chercheur sélectionné.")
        return 1
    blocked = False
    if not args.enrich_only:
        log.info("Backend : %s | max %d publications/chercheur | délai %s–%s s | resume=%s",
                 collector.backend.name, max_pubs, settings["scraping"]["min_delay"], settings["scraping"]["max_delay"], args.resume)
        summary = collector.run(researchers, max_pubs, resume=args.resume)
        blocked = summary["blocked"]
        print("\n=== Résumé de la collecte Scholar ===")
        print(f"Traités : {summary['processed']} | déjà faits (ignorés) : {summary['skipped_done']}")
        print(f"Statuts de profil : {summary['profile_status_counts']}")
        n_pubs = sum(len(s['publications']) for s in collector.states.values())
        print(f"Publications brutes collectées (cumul) : {n_pubs}")
        if blocked:
            print(f"⚠ BLOCAGE Google Scholar à {summary['blocked_at']} : données sauvegardées. "
                  "Relancez plus tard avec la même commande (reprise automatique). Aucun contournement n'est tenté.")
    if settings["enrichment"]["enabled"] and not args.skip_enrichment:
        run_enrichment(collector, settings, log)
    else:
        collector.write_consolidated()
    review = resolve_path(settings, "raw_dir") / "scholar_candidates_review.csv"
    print(f"\nProfils à vérifier manuellement (ambigus / non trouvés) : {review}")
    print("Pour valider un profil : ajoutez une ligne « chercheur_id,scholar_id » dans "
          f"{settings['scraping']['overrides_file']} puis relancez avec --resume.")
    return 2 if blocked else 0


if __name__ == "__main__":
    sys.exit(main())
