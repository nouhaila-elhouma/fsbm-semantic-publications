"""Enrichissement des publications : DOI et abstracts via des API académiques publiques.

Ordre des sources (configurable, ``enrichment.sources``) :
  0. « Description » de Google Scholar (déjà collectée) — conservée si complète ;
  1. Crossref ; 2. OpenAlex ; 3. Semantic Scholar ;
  4. (optionnel, désactivé par défaut) métadonnées ``citation_abstract`` de la page DOI de l'éditeur.

Garde-fous : un résultat trouvé par recherche de titre n'est retenu que si le titre
correspond (seuil configurable) et si l'année est cohérente ; RIEN n'est jamais fabriqué —
sans source fiable, l'abstract reste ``None`` avec un statut explicite.
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any, Callable, Optional
from urllib.parse import quote

import requests
from bs4 import BeautifulSoup

from src.preprocessing.text_cleaner import (clean_text_light, extract_doi, is_truncated, looks_like_affiliation,
                                            normalize_doi, normalize_for_matching, strip_html, title_similarity)
from src.utils.io import read_json, write_json
from src.utils.retry import TransientError, polite_sleep, retry

logger = logging.getLogger(__name__)
CACHE_VERSION = 2       # v2 : + revue/éditeur ; les entrées v1 sont ré-enrichies
MIN_ABSTRACT_CHARS = 80  # en dessous, une « description » d'API est rarement un vrai abstract


# --------------------------------------------------------------------------- helpers purs
def openalex_abstract(inverted_index: Optional[dict[str, list[int]]]) -> Optional[str]:
    """Reconstruit l'abstract OpenAlex à partir de ``abstract_inverted_index`` (mot → positions)."""
    if not inverted_index:
        return None
    positions: dict[int, str] = {}
    for word, idxs in inverted_index.items():
        for idx in idxs:
            positions[idx] = word
    return " ".join(positions[i] for i in sorted(positions)) or None


def clean_api_abstract(text: Optional[str]) -> Optional[str]:
    """Nettoie un abstract d'API (JATS/HTML) en conservant la casse."""
    cleaned = clean_text_light(strip_html(text)) if text else None
    if cleaned and cleaned.lower().startswith("abstract"):
        cleaned = cleaned[8:].lstrip(" :.-–—") or None
    return None if looks_like_affiliation(cleaned) else cleaned      # liste d'affiliations ≠ abstract


def year_compatible(year_a: Optional[int], year_b: Optional[int], tolerance: int = 1) -> bool:
    """Vrai si l'une des années est inconnue ou si l'écart est ≤ ``tolerance``."""
    return year_a is None or year_b is None or abs(year_a - year_b) <= tolerance


def best_title_match(pub: dict[str, Any], items: list[dict[str, Any]], title_of: Callable[[dict], Optional[str]],
                     year_of: Callable[[dict], Optional[int]], threshold: float) -> Optional[dict[str, Any]]:
    """Retourne le résultat d'API dont le titre correspond le mieux (au-dessus du seuil), sinon ``None``."""
    best, best_score = None, 0.0
    for item in items:
        score = title_similarity(pub.get("titre"), title_of(item))
        if score >= threshold and year_compatible(pub.get("annee"), year_of(item)) and score > best_score:
            best, best_score = item, score
    return best


def cache_key(pub: dict[str, Any]) -> str:
    """Clé de cache indépendante du chercheur (un article partagé n'est interrogé qu'une fois)."""
    return hashlib.sha1(f"{normalize_for_matching(pub.get('titre'))}|{pub.get('annee')}".encode()).hexdigest()[:16]


def _first(values: Any) -> Optional[str]:
    if isinstance(values, list):
        return values[0] if values else None
    return values


# --------------------------------------------------------------------------- enrichisseur
class PublicationEnricher:
    """Complète DOI + abstract de chaque publication brute, avec cache et reprise."""

    def __init__(self, settings: dict[str, Any], cache_path: Path, contact_email: Optional[str] = None,
                 s2_api_key: Optional[str] = None, session: Optional[requests.Session] = None,
                 delay_fn: Optional[Callable[[], Any]] = None) -> None:
        self.cfg = settings["enrichment"]
        self.cache_path = Path(cache_path)
        self.cache: dict[str, dict[str, Any]] = read_json(self.cache_path, default={})
        self.session = session or requests.Session()
        ua = settings["scraping"]["user_agent"] + (f" mailto:{contact_email}" if contact_email else "")
        self.session.headers.update({"User-Agent": ua})
        self.contact_email = contact_email
        self.s2_api_key = s2_api_key
        self.delay = delay_fn or (lambda: polite_sleep(self.cfg["min_delay"], self.cfg["max_delay"]))
        self.threshold = self.cfg["title_match_threshold"]
        self._get_json = retry(self.cfg["retries"], 2.0, retry_on=(TransientError,))(self._get_json_once)

    # ------------------------------------------------------------------ HTTP
    def _get_json_once(self, url: str, params: Optional[dict[str, Any]] = None,
                       headers: Optional[dict[str, str]] = None) -> Optional[dict[str, Any]]:
        try:
            resp = self.session.get(url, params=params, headers=headers, timeout=self.cfg["timeout"])
        except requests.RequestException as exc:
            raise TransientError(f"réseau : {exc}") from exc
        if resp.status_code == 404:
            return None
        if resp.status_code == 429 or resp.status_code >= 500:
            raise TransientError(f"HTTP {resp.status_code} sur {url.split('?')[0]}")
        resp.raise_for_status()
        return resp.json()

    def _call(self, url: str, params: Optional[dict[str, Any]] = None, headers: Optional[dict[str, str]] = None):
        self.delay()
        return self._get_json(url, params, headers)

    # ------------------------------------------------------------------ sources
    def _crossref(self, pub: dict[str, Any], doi: Optional[str]) -> dict[str, Any]:
        if doi:
            data = self._call(f"https://api.crossref.org/works/{quote(doi, safe='/')}")
            item = data.get("message") if data else None
        else:
            params = {"query.bibliographic": pub["titre"], "rows": 5, "select": "DOI,title,abstract,issued,container-title,publisher"}
            if self.contact_email:
                params["mailto"] = self.contact_email
            data = self._call("https://api.crossref.org/works", params)
            items = (data or {}).get("message", {}).get("items", [])
            item = best_title_match(pub, items, lambda i: _first(i.get("title")),
                                    lambda i: _first(_first((i.get("issued") or {}).get("date-parts")) or [None]),
                                    self.threshold)
        if not item:
            return {}
        return {"doi": normalize_doi(item.get("DOI")), "abstract": clean_api_abstract(item.get("abstract")),
                "journal": clean_text_light(_first(item.get("container-title"))), "publisher": clean_text_light(item.get("publisher"))}

    def _openalex(self, pub: dict[str, Any], doi: Optional[str]) -> dict[str, Any]:
        params = {"mailto": self.contact_email} if self.contact_email else {}
        if doi:
            item = self._call(f"https://api.openalex.org/works/https://doi.org/{quote(doi, safe='/')}", params)
        else:
            data = self._call("https://api.openalex.org/works", {**params, "search": pub["titre"], "per-page": 5})
            item = best_title_match(pub, (data or {}).get("results", []), lambda i: i.get("title"),
                                    lambda i: i.get("publication_year"), self.threshold)
        if not item:
            return {}
        source = ((item.get("primary_location") or {}).get("source")) or {}
        return {"doi": normalize_doi(item.get("doi")),
                "abstract": clean_api_abstract(openalex_abstract(item.get("abstract_inverted_index"))),
                "journal": clean_text_light(source.get("display_name")),
                "publisher": clean_text_light(source.get("host_organization_name"))}

    def _semantic_scholar(self, pub: dict[str, Any], doi: Optional[str]) -> dict[str, Any]:
        headers = {"x-api-key": self.s2_api_key} if self.s2_api_key else None
        fields = "title,abstract,year,externalIds,venue"
        if doi:
            item = self._call(f"https://api.semanticscholar.org/graph/v1/paper/DOI:{quote(doi, safe='/')}",
                              {"fields": fields}, headers)
        else:
            data = self._call("https://api.semanticscholar.org/graph/v1/paper/search/match",
                              {"query": pub["titre"], "fields": fields}, headers)
            item = best_title_match(pub, (data or {}).get("data", []), lambda i: i.get("title"),
                                    lambda i: i.get("year"), self.threshold)
        if not item:
            return {}
        return {"doi": normalize_doi((item.get("externalIds") or {}).get("DOI")),
                "abstract": clean_api_abstract(item.get("abstract")), "journal": clean_text_light(item.get("venue"))}

    def _publisher_page(self, doi: str) -> dict[str, Any]:
        """Métadonnées ``citation_abstract`` de la page DOI (uniquement si l'éditeur les expose)."""
        self.delay()
        try:
            resp = self.session.get(f"https://doi.org/{quote(doi, safe='/')}", timeout=self.cfg["timeout"],
                                    headers={"Accept": "text/html"})
        except requests.RequestException as exc:
            raise TransientError(str(exc)) from exc
        if resp.status_code != 200:
            raise TransientError(f"page éditeur HTTP {resp.status_code}")
        soup = BeautifulSoup(resp.text, "html.parser")
        for name in ("citation_abstract", "dc.description", "DC.Description"):
            tag = soup.find("meta", attrs={"name": name})
            text = clean_api_abstract(tag.get("content")) if tag and tag.get("content") else None
            if text and len(text) >= 150:
                return {"abstract": text}
        return {}

    # ------------------------------------------------------------------ orchestration
    def _run_source(self, name: str, pub: dict[str, Any], doi: Optional[str]) -> tuple[str, dict[str, Any]]:
        runner = {"crossref": self._crossref, "openalex": self._openalex, "semantic_scholar": self._semantic_scholar}[name]
        try:
            result = runner(pub, doi)
        except (TransientError, requests.RequestException, ValueError) as exc:
            logger.warning("Source %s indisponible pour « %s » : %s", name, (pub["titre"] or "")[:50], exc)
            return "api_error", {}
        return ("found" if result.get("abstract") else "not_found"), result

    def enrich_publication(self, pub: dict[str, Any]) -> bool:
        """Complète ``pub`` en place. Retourne ``True`` si l'enrichissement est définitif (pas d'erreur d'API)."""
        attempts: list[dict[str, str]] = []
        doi = normalize_doi(pub.get("doi")) or extract_doi(pub.get("external_url")) or extract_doi(pub.get("pdf_url"))
        has_full = pub.get("abstract_status") == "found"
        api_error = False
        found_journal: Optional[str] = None
        found_publisher: Optional[str] = None
        has_venue = bool(pub.get("journal") or pub.get("conference"))

        for source in self.cfg["sources"]:
            if has_full and doi and (has_venue or found_journal):
                break
            status, result = self._run_source(source, pub, doi)
            attempts.append({"source": source, "status": status})
            api_error |= status == "api_error"
            doi = doi or result.get("doi")
            found_journal = found_journal or result.get("journal")
            found_publisher = found_publisher or result.get("publisher")
            abstract = result.get("abstract")
            current = pub.get("abstract") or ""
            if abstract and not is_truncated(abstract) and len(abstract) >= MIN_ABSTRACT_CHARS and not has_full:
                if not current or len(abstract) > len(current):
                    pub.update(abstract=abstract, abstract_source=source, abstract_status="found")
                    has_full = True

        if not has_full and doi and self.cfg.get("publisher_page"):
            try:
                page = self._publisher_page(doi)
                attempts.append({"source": "publisher_page", "status": "found" if page else "not_found"})
                if page.get("abstract"):
                    pub.update(abstract=page["abstract"], abstract_source="publisher_page", abstract_status="found")
                    has_full = True
            except TransientError as exc:
                attempts.append({"source": "publisher_page", "status": "publisher_unavailable"})
                logger.info("Page éditeur inaccessible (%s)", exc)

        pub["doi"] = doi
        if found_journal and not has_venue:
            pub["journal"] = found_journal                      # revue/actes : Crossref > OpenAlex > Semantic Scholar
        if found_publisher and not pub.get("publisher"):
            pub["publisher"] = found_publisher
        pub["abstract_attempts"] = attempts
        if not has_full:
            if pub.get("abstract"):
                pub["abstract_status"] = "truncated" if is_truncated(pub["abstract"]) else "found"
            elif api_error:
                pub["abstract_status"] = "api_error"
            elif any(a["status"] == "publisher_unavailable" for a in attempts):
                pub["abstract_status"] = "publisher_unavailable"
            else:
                pub["abstract_status"] = "not_found"
        return not api_error

    def enrich_all(self, publications: list[dict[str, Any]], resume: bool = True, save_every: int = 10) -> dict[str, int]:
        """Enrichit toutes les publications ; les résultats définitifs sont mis en cache (reprise)."""
        stats = {"total": len(publications), "from_cache": 0, "enriched": 0, "api_errors": 0}
        fields = ("doi", "abstract", "abstract_source", "abstract_status", "abstract_attempts", "journal", "publisher")
        for i, pub in enumerate(publications, start=1):
            if not pub.get("titre"):
                continue
            key = cache_key(pub)
            cached = self.cache.get(key) if resume else None
            if cached and cached.get("final") and cached.get("v") == CACHE_VERSION:
                pub.update({f: cached[f] for f in fields if f not in ("journal", "publisher") or not pub.get(f)})
                stats["from_cache"] += 1
                continue
            final = self.enrich_publication(pub)
            stats["enriched"] += 1
            stats["api_errors"] += 0 if final else 1
            self.cache[key] = {**{f: pub.get(f) for f in fields}, "final": final, "v": CACHE_VERSION}
            if i % save_every == 0:
                write_json(self.cache_path, self.cache)
                logger.info("Enrichissement : %d/%d publications", i, len(publications))
        write_json(self.cache_path, self.cache)
        return stats
