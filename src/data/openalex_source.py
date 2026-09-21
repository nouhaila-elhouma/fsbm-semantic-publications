"""Source de repli OpenAlex (API officielle, ouverte) quand Google Scholar bloque la collecte.

Pourquoi : Scholar renvoie des HTTP 429 (limitation d'accès) ; le projet ne les contourne jamais (pas de proxy, pas de
changement d'IP). OpenAlex fournit, sans limitation gênante, publications, abstracts complets, DOI, revues, citations
et h-index/i10-index par auteur. Les données de repli sont TOUJOURS étiquetées ``data_source = "openalex"`` ; si Scholar
se débloque plus tard, ses données reprennent automatiquement la priorité (voir ``ScholarCollector.effective_states``).

Limites assumées (documentées dans le README) :
  * OpenAlex fragmente parfois un auteur en plusieurs profils : on fusionne ceux dont le nom est quasi identique ET
    qui sont rattachés à l'Université Hassan II de Casablanca ;
  * ses métriques (citations, h-index, i10-index) ne sont pas celles de Google Scholar (couverture différente) ;
  * un chercheur sans profil OpenAlex rattaché à Hassan II reste sans données de repli (jamais deviné).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Optional

import requests

from src.data.publication_enricher import openalex_abstract
from src.data.scholar_matching import name_match_score
from src.data.scholar_scraper import new_publication_record
from src.data.scholar_parsing import profile_url
from src.preprocessing.text_cleaner import clean_text_light, looks_like_affiliation, normalize_doi
from src.utils.io import read_json, write_json
from src.utils.retry import TransientError, polite_sleep, retry

logger = logging.getLogger(__name__)
API = "https://api.openalex.org"
HASSAN_II_INSTITUTION = "I99297268"      # « University of Hassan II Casablanca » (vérifié via l'API /institutions)
WORK_FIELDS = ("id,display_name,publication_year,publication_date,doi,cited_by_count,authorships,primary_location,"
               "best_oa_location,abstract_inverted_index,type")


def _short_id(url: str) -> str:
    return url.rsplit("/", 1)[-1]


class OpenAlexSource:
    """Client OpenAlex minimal : recherche d'auteurs, sélection prudente, récupération des publications."""

    name = "openalex"

    def __init__(self, contact_email: Optional[str] = None, timeout: float = 30, retries: int = 4,
                 delay_fn: Optional[Callable[[], Any]] = None, session: Optional[requests.Session] = None,
                 institution_id: str = HASSAN_II_INSTITUTION, min_score: float = 0.90, api_key: Optional[str] = None) -> None:
        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": "FSBM-Semantic-Publications/1.0 (academic research project)"})
        self.contact_email, self.timeout, self.api_key = contact_email, timeout, api_key
        self.institution_id, self.min_score = institution_id, min_score
        self._delay = delay_fn or (lambda: polite_sleep(0.2, 0.6))
        self._get_json = retry(retries, 3.0, retry_on=(TransientError,))(self._get_once)

    def _get_once(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        params = {**params, **({"mailto": self.contact_email} if self.contact_email else {}),
                  **({"api_key": self.api_key} if self.api_key else {})}
        try:
            resp = self.session.get(f"{API}/{path}", params=params, timeout=self.timeout)
        except requests.RequestException as exc:
            raise TransientError(f"réseau : {exc}") from exc
        if resp.status_code == 429 or resp.status_code >= 500:
            raise TransientError(f"HTTP {resp.status_code} sur {path}")
        resp.raise_for_status()
        return resp.json()

    def _call(self, path: str, **params: Any) -> dict[str, Any]:
        self._delay()
        return self._get_json(path, params)

    # ------------------------------------------------------------------ auteurs
    def search_authors(self, name: str, per_page: int = 15) -> list[dict[str, Any]]:
        return self._call("authors", search=name, **{"per-page": per_page}).get("results", [])

    def has_institution(self, author: dict[str, Any]) -> bool:
        """Auteur rattaché à l'Université Hassan II (dernière institution connue OU historique d'affiliations)."""
        ids = [_short_id(i["id"]) for i in author.get("last_known_institutions") or []]
        ids += [_short_id(a["institution"]["id"]) for a in author.get("affiliations") or [] if a.get("institution")]
        return self.institution_id in ids

    def choose_authors(self, names: list[str], candidates: list[dict[str, Any]]) -> dict[str, Any]:
        """Sélection prudente : nom quasi identique + rattachement Hassan II ; fragments fusionnés ; homonymes écartés."""
        scored = []
        for cand in candidates:
            score = max(name_match_score(n, cand.get("display_name")) for n in names)
            if score >= self.min_score and self.has_institution(cand):
                scored.append((score, cand))
        if not scored:
            return {"status": "not_found", "detail": "aucun profil OpenAlex de ce nom rattaché à l'Université Hassan II"}
        scored.sort(key=lambda t: t[1].get("works_count", 0), reverse=True)
        primary_score, primary = scored[0]
        big = [c for s, c in scored[1:] if c.get("works_count", 0) >= max(20, 0.5 * primary.get("works_count", 0))]
        if big:                                            # deux « gros » profils de même nom : homonymes possibles
            return {"status": "ambiguous", "detail": f"plusieurs profils importants ({primary['display_name']}, {big[0]['display_name']})"}
        fragments = [c for s, c in scored[1:] if s >= 0.93]
        return {"status": "matched", "primary": primary, "fragments": fragments, "name_score": primary_score}

    # ------------------------------------------------------------------ publications
    def fetch_works(self, author_ids: list[str], max_publications: int) -> list[dict[str, Any]]:
        params = {"filter": "author.id:" + "|".join(author_ids), "sort": "cited_by_count:desc",
                  "per-page": min(max(max_publications, 1), 200), "select": WORK_FIELDS}
        return self._call("works", **params).get("results", [])[:max_publications]

    def work_to_record(self, chercheur_id: str, scholar_id: Optional[str], work: dict[str, Any]) -> dict[str, Any]:
        source = ((work.get("primary_location") or {}).get("source")) or {}
        row = {"scholar_pub_id": f"oa:{_short_id(work['id'])}", "titre": clean_text_light(work.get("display_name")),
               "auteurs_listing": None, "venue_raw": source.get("display_name"), "annee": work.get("publication_year"),
               "citations": work.get("cited_by_count"), "scholar_url": None}
        rec = new_publication_record({"chercheur_id": chercheur_id, "scholar_id": scholar_id}, row, fetch_details=False)
        abstract = openalex_abstract(work.get("abstract_inverted_index"))
        if abstract and looks_like_affiliation(abstract):
            abstract = None
        is_conf = source.get("type") == "conference"
        rec.update(
            auteurs=[clean_text_light((a.get("author") or {}).get("display_name")) for a in work.get("authorships") or []
                     if (a.get("author") or {}).get("display_name")],
            auteurs_truncated=False, date_publication_raw=work.get("publication_date"),
            journal=None if is_conf else clean_text_light(source.get("display_name")),
            conference=clean_text_light(source.get("display_name")) if is_conf else None,
            publisher=clean_text_light(source.get("host_organization_name")), doi=normalize_doi(work.get("doi")),
            pdf_url=(work.get("best_oa_location") or {}).get("pdf_url"), abstract=abstract,
            abstract_source="openalex" if abstract else None, abstract_status="found" if abstract else "not_found",
            detail_status="skipped", scrape_status="openalex", data_source="openalex")
        return rec

    # ------------------------------------------------------------------ chercheur complet
    def collect(self, rec: dict[str, Any], names: list[str], scholar_id: Optional[str], max_publications: int = 50) -> dict[str, Any]:
        """État d'un chercheur au format des états Scholar (mêmes clés), avec ``data_source = "openalex"``."""
        state = {k: rec.get(k) for k in ("chercheur_id", "nom_complet", "etablissement", "laboratoire", "equipe", "type_membre")}
        state.update(scholar_profile_status="pending", status_detail=None, profile_match_confidence=None,
                     match_method="openalex_name_institution", candidates=[], scholar_id=scholar_id,
                     scholar_url=profile_url(scholar_id) if scholar_id else None, profile=None, coauthor_hits=[],
                     publications=[], collection_status="n/a", backend="openalex", data_source="openalex",
                     openalex_ids=[])
        candidates: dict[str, dict[str, Any]] = {}
        for name in dict.fromkeys(n for n in names if n):
            for cand in self.search_authors(name):
                candidates.setdefault(cand["id"], cand)
        choice = self.choose_authors(names, list(candidates.values()))
        state["candidates"] = [{"openalex_id": _short_id(c["id"]), "name": c.get("display_name"), "works_count": c.get("works_count")}
                               for c in list(candidates.values())[:8]]
        if choice["status"] != "matched":
            state.update(scholar_profile_status=choice["status"], status_detail=f"openalex : {choice['detail']}")
            return state
        primary, fragments = choice["primary"], choice["fragments"]
        ids = [_short_id(primary["id"])] + [_short_id(f["id"]) for f in fragments]
        works = self.fetch_works(ids, max_publications)
        stats = primary.get("summary_stats") or {}
        institutions = list(dict.fromkeys(i["display_name"] for i in primary.get("last_known_institutions") or []))
        state.update(
            scholar_profile_status="matched", collection_status="complete", openalex_ids=ids,
            profile_match_confidence=round(0.6 * choice["name_score"] + 0.4, 3),
            status_detail="Google Scholar limite l'accès (HTTP 429) : données OpenAlex (profil rattaché à l'Université Hassan II)",
            profile={"name": primary.get("display_name"), "affiliation": ", ".join(institutions) or None, "email_domain": None,
                     "interests": [t["display_name"] for t in (primary.get("topics") or [])[:5]],
                     "metrics": {"citations": primary.get("cited_by_count"), "h_index": stats.get("h_index"),
                                 "i10_index": stats.get("i10_index")}, "since_year": None, "coauthors": []},
            publications=[self.work_to_record(rec["chercheur_id"], scholar_id, w) for w in works])
        return state


class OpenAlexFallback:
    """Orchestration + cache par chercheur (``data/raw/cache/openalex/<id>.json``), reprise incluse."""

    def __init__(self, source: OpenAlexSource, cache_dir: Path) -> None:
        self.source = source
        self.dir = Path(cache_dir) / "openalex"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.states: dict[str, dict[str, Any]] = {}
        for path in sorted(self.dir.glob("*.json")):
            state = read_json(path)
            if state and "chercheur_id" in state:
                self.states[state["chercheur_id"]] = state

    def save(self, state: dict[str, Any]) -> None:
        write_json(self.dir / f"{state['chercheur_id']}.json", state)
        self.states[state["chercheur_id"]] = state

    def run(self, researchers: list[dict[str, Any]], alt_names: dict[str, list[str]], scholar_ids: dict[str, str],
            max_publications: int = 50, resume: bool = True) -> dict[str, int]:
        counts = {"matched": 0, "not_found": 0, "ambiguous": 0, "error": 0, "skipped_done": 0}
        for i, rec in enumerate(researchers, start=1):
            cid = rec["chercheur_id"]
            if resume and cid in self.states and self.states[cid].get("scholar_profile_status") != "error":
                counts["skipped_done"] += 1
                continue
            names = [rec["nom_complet"], *alt_names.get(cid, [])]
            logger.info("OpenAlex %d/%d : %s", i, len(researchers), rec["nom_complet"])
            try:
                state = self.source.collect(rec, names, scholar_ids.get(cid), max_publications)
            except Exception as exc:  # noqa: BLE001 — une erreur sur un chercheur n'arrête pas les autres
                logger.error("OpenAlex : erreur pour %s : %s", rec["nom_complet"], exc)
                state = {"chercheur_id": cid, "nom_complet": rec["nom_complet"], "scholar_profile_status": "error",
                         "status_detail": f"openalex : {type(exc).__name__}: {exc}", "publications": [], "data_source": "openalex"}
            self.save(state)
            counts[state["scholar_profile_status"]] = counts.get(state["scholar_profile_status"], 0) + 1
        return counts
