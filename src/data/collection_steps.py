"""Étapes partagées entre les scripts de collecte (02 : Scholar, 02b : repli OpenAlex)."""
from __future__ import annotations

import logging
from typing import Any

from src.data.publication_enricher import PublicationEnricher
from src.data.scholar_scraper import ScholarCollector
from src.utils.config import get_env, resolve_path


def run_enrichment(collector: ScholarCollector, settings: dict[str, Any], log: logging.Logger) -> dict[str, int]:
    """Enrichit (DOI, abstracts, revue) les publications retenues — Scholar prioritaire, sinon repli OpenAlex —
    puis régénère ``scholars_raw.json`` / ``publications_raw.json``.
    """
    enricher = PublicationEnricher(settings, resolve_path(settings, "cache_dir") / "enrichment_cache.json",
                                   contact_email=get_env("CONTACT_EMAIL"), s2_api_key=get_env("SEMANTIC_SCHOLAR_API_KEY"),
                                   openalex_api_key=get_env("OPENALEX_API_KEY"))
    if not get_env("CONTACT_EMAIL"):
        log.warning("CONTACT_EMAIL absent : Crossref/OpenAlex fonctionnent mais le « polite pool » est recommandé (.env).")
    effective = collector.effective_states()
    all_pubs = [p for state in effective.values() for p in state.get("publications", [])]
    log.info("Enrichissement de %d publications (Crossref → OpenAlex → Semantic Scholar)…", len(all_pubs))
    stats = enricher.enrich_all(all_pubs)
    for state in effective.values():
        collector.save_any(state)
    collector.write_consolidated()
    log.info("Enrichissement terminé : %s", stats)
    return stats
