"""Collecte Google Scholar : backends interchangeables + collecteur robuste.

Garanties de robustesse :
  * délai aléatoire poli avant chaque requête, backoff exponentiel, retries limités ;
  * AUCUN contournement de CAPTCHA/blocage : ``BlockedError`` => sauvegarde puis arrêt propre ;
  * cache disque par chercheur (``data/raw/cache/scholar/<chercheur_id>.json``) ;
  * checkpoint après chaque chercheur (et toutes les N publications) ;
  * reprise exacte là où l'exécution s'est arrêtée (``resume=True``) ;
  * une erreur sur un chercheur/une publication n'arrête pas les autres et ne perd rien.
"""
from __future__ import annotations

import csv
import hashlib
import itertools
import logging
import re
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

import requests

from src.data.robots import RobotsDisallowedError, RobotsRules, SearchNotAllowedError
from src.data.scholar_matching import MatchDecision, coauthor_bonus, decide_match
from src.data.scholar_parsing import (BASE_URL, ScholarParseError, detect_block, parse_profile,
                                      parse_publication_detail, parse_search_results, profile_url)
from src.preprocessing.text_cleaner import is_truncated
from src.utils.io import read_json, write_json
from src.utils.retry import BlockedError, TransientError, polite_sleep, retry

logger = logging.getLogger(__name__)
DETAIL_CHECKPOINT_EVERY = 5


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# =========================================================================== backends
class ScholarBackend(ABC):
    """Interface commune : changer de méthode de collecte = changer de classe (config: scraping.backend)."""

    name: str = "abstract"
    search_allowed: bool = True      # False : la recherche d'auteurs par nom est désactivée (robots.txt)

    @abstractmethod
    def search_authors(self, query: str) -> list[dict[str, Any]]:
        """Candidats : ``scholar_id, name, affiliation, email_domain, interests, cited_by, scholar_url``."""

    @abstractmethod
    def fetch_author(self, scholar_id: str, max_publications: int) -> dict[str, Any]:
        """Profil + liste de publications (même forme que ``parse_profile``)."""

    @abstractmethod
    def fetch_publication_detail(self, scholar_id: str, scholar_pub_id: str) -> dict[str, Any]:
        """Métadonnées complètes d'une publication (même forme que ``parse_publication_detail``)."""


class HttpScholarBackend(ScholarBackend):
    """requests + BeautifulSoup sur les pages publiques /citations. Aucun contournement de protection.

    Conformité ``robots.txt`` (activée par défaut) : chaque URL est vérifiée avant envoi.
    Le robots.txt de Google Scholar interdit ``/citations?`` sauf ``/citations?user=`` (profils) et interdit
    ``cstart=`` (pagination). En conséquence :
      * les URL de profil et de publication sont construites avec ``user=`` en premier ;
      * la pagination est désactivée (au plus ``pagesize`` ≤ 100 publications par chercheur) ;
      * la recherche d'auteurs par nom est REFUSÉE sauf opt-in explicite ``allow_search=True``
        (``--allow-author-search``), à assumer par l'utilisateur ; sinon fournir les Scholar ID à la main.
    """

    name = "http"
    ROBOTS_URL = f"{BASE_URL}/robots.txt"

    def __init__(self, user_agent: str, timeout: float = 30, retries: int = 3, backoff_base: float = 5,
                 delay_fn: Optional[Callable[[], Any]] = None, session: Optional[requests.Session] = None,
                 min_delay: float = 4, max_delay: float = 9, respect_robots: bool = True,
                 allow_search: bool = False, robots_text: Optional[str] = None) -> None:
        self.timeout = timeout
        self.respect_robots, self.allow_search = respect_robots, allow_search
        self.search_allowed = (not respect_robots) or allow_search
        self._session = session or requests.Session()
        self._session.headers.update({"User-Agent": user_agent, "Accept-Language": "en"})
        self._delay = delay_fn or (lambda: polite_sleep(min_delay, max_delay))
        self._n_requests = 0
        self._rules: Optional[RobotsRules] = RobotsRules.from_text(robots_text) if robots_text is not None else None
        self._warned_search = False
        self._get_with_retry = retry(retries, backoff_base, retry_on=(TransientError,))(self._get_once)

    # ------------------------------------------------------------------ robots.txt
    def _robots(self) -> RobotsRules:
        if self._rules is None:
            try:
                resp = self._session.get(self.ROBOTS_URL, timeout=self.timeout)
            except requests.RequestException as exc:
                raise TransientError(f"robots.txt inaccessible : {exc}") from exc
            if resp.status_code != 200:
                # échec « fermé » : sans robots.txt lisible on n'envoie rien
                raise RobotsDisallowedError(f"robots.txt illisible (HTTP {resp.status_code}) : requêtes suspendues")
            self._rules = RobotsRules.from_text(resp.text)
        return self._rules

    def _check_allowed(self, params: dict[str, Any], is_search: bool = False) -> None:
        if not self.respect_robots:
            return
        target = requests.Request("GET", f"{BASE_URL}/citations", params=params).prepare().path_url
        if is_search and self.allow_search:
            if not self._warned_search:
                logger.warning("Recherche d'auteurs ACTIVÉE par l'utilisateur alors que robots.txt de Google Scholar "
                               "l'interdit (Disallow: /citations?). Usage sous votre responsabilité.")
                self._warned_search = True
            return
        if is_search:
            raise SearchNotAllowedError(
                "recherche d'auteurs par nom désactivée (interdite par robots.txt de Google Scholar). "
                "Renseignez les Scholar ID dans scholar_overrides.csv, ou activez explicitement --allow-author-search.")
        if not self._robots().is_allowed(target):
            raise RobotsDisallowedError(f"URL interdite par robots.txt, non envoyée : {target}")

    # ------------------------------------------------------------------ requêtes
    def _get_once(self, params: dict[str, Any]) -> str:
        try:
            resp = self._session.get(f"{BASE_URL}/citations", params=params, timeout=self.timeout)
        except requests.RequestException as exc:
            raise TransientError(f"erreur réseau : {exc}") from exc
        detect_block(resp.status_code, resp.url, resp.text)
        if resp.status_code >= 500:
            raise TransientError(f"HTTP {resp.status_code}")
        if resp.status_code == 404:
            raise ScholarParseError(f"HTTP 404 pour {resp.url}")
        resp.raise_for_status()
        return resp.text

    def _get(self, params: dict[str, Any], is_search: bool = False) -> str:
        self._check_allowed(params, is_search)
        if self._n_requests:
            self._delay()
        self._n_requests += 1
        return self._get_with_retry(params)

    def search_authors(self, query: str) -> list[dict[str, Any]]:
        html = self._get({"view_op": "search_authors", "mauthors": query, "hl": "en"}, is_search=True)
        return parse_search_results(html)

    def fetch_author(self, scholar_id: str, max_publications: int) -> dict[str, Any]:
        pagesize = max(1, min(100, max_publications))
        first: Optional[dict[str, Any]] = None
        publications: list[dict[str, Any]] = []
        cstart = 0
        while True:
            params: dict[str, Any] = {"user": scholar_id, "hl": "en", "pagesize": pagesize}
            if cstart:                                    # pagination : « Disallow: /citations?*cstart= »
                params["cstart"] = cstart
            parsed = parse_profile(self._get(params))
            first = first or parsed
            publications.extend(parsed["publications"])
            if not parsed["has_more"] or not parsed["publications"] or len(publications) >= max_publications:
                break
            if self.respect_robots:
                logger.info("Pagination interdite par robots.txt : limité aux %d publications de la première page.",
                            len(publications))
                break
            cstart += pagesize
        first["publications"] = publications[:max_publications]
        return first

    def fetch_publication_detail(self, scholar_id: str, scholar_pub_id: str) -> dict[str, Any]:
        # « user= » en premier : l'URL relève de « Allow: /citations?user= » (pas de « view_op » en tête)
        html = self._get({"user": scholar_id, "view_op": "view_citation", "hl": "en",
                          "citation_for_view": scholar_pub_id})
        return parse_publication_detail(html)


class ScholarlyBackend(ScholarBackend):
    """Backend basé sur la bibliothèque ``scholarly`` (sans proxy : aucun contournement)."""

    name = "scholarly"

    def __init__(self, delay_fn: Optional[Callable[[], Any]] = None, min_delay: float = 4, max_delay: float = 9) -> None:
        from scholarly import scholarly  # import local : paquet lourd, optionnel selon le backend

        self._s = scholarly
        self._delay = delay_fn or (lambda: polite_sleep(min_delay, max_delay))
        self._pubs: dict[str, dict[str, Any]] = {}

    def _guard(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        self._delay()
        try:
            return func(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 — on traduit les erreurs scholarly connues
            if type(exc).__name__ in {"MaxTriesExceededException", "DOSException"}:
                raise BlockedError(f"scholarly : {type(exc).__name__} — accès limité") from exc
            if isinstance(exc, (requests.RequestException, ConnectionError, TimeoutError)):
                raise TransientError(str(exc)) from exc
            raise

    def search_authors(self, query: str) -> list[dict[str, Any]]:
        results = self._guard(lambda: list(itertools.islice(self._s.search_author(query), 10)))
        return [{
            "scholar_id": r.get("scholar_id"), "name": r.get("name"), "affiliation": r.get("affiliation"),
            "email_domain": (r.get("email_domain") or "").lstrip("@").lower() or None,
            "interests": list(r.get("interests") or []), "cited_by": r.get("citedby"),
            "scholar_url": profile_url(r["scholar_id"]),
        } for r in results if r.get("scholar_id")]

    def fetch_author(self, scholar_id: str, max_publications: int) -> dict[str, Any]:
        def _fill() -> dict[str, Any]:
            author = self._s.search_author_id(scholar_id)
            return self._s.fill(author, sections=["basics", "indices", "coauthors", "publications"],
                                publication_limit=max_publications)

        author = self._guard(_fill)
        publications = []
        for pub in (author.get("publications") or [])[:max_publications]:
            pub_id = pub.get("author_pub_id")
            self._pubs[pub_id] = pub
            bib = pub.get("bib", {})
            publications.append({
                "scholar_pub_id": pub_id, "titre": bib.get("title"), "auteurs_listing": None,
                "venue_raw": bib.get("citation"), "annee": _to_year(bib.get("pub_year")),
                "citations": pub.get("num_citations"),
                "scholar_url": f"{BASE_URL}/citations?view_op=view_citation&hl=en&citation_for_view={pub_id}",
            })
        # scholarly remplace certaines métriques « 5 ans » manquantes par 0 : on ne les reprend donc pas
        # (None = inconnu) plutôt que de risquer une valeur fausse.
        metrics = {"citations": author.get("citedby"), "h_index": author.get("hindex"),
                   "i10_index": author.get("i10index")}
        return {"name": author.get("name"), "affiliation": author.get("affiliation"),
                "email_domain": (author.get("email_domain") or "").lstrip("@").lower() or None,
                "interests": list(author.get("interests") or []), "metrics": metrics, "since_year": None,
                "coauthors": [c.get("name") for c in author.get("coauthors", []) if c.get("name")],
                "publications": publications, "has_more": False}

    def fetch_publication_detail(self, scholar_id: str, scholar_pub_id: str) -> dict[str, Any]:
        from scholarly.data_types import PublicationSource

        pub = self._pubs.get(scholar_pub_id) or {
            "container_type": "Publication", "source": PublicationSource.AUTHOR_PUBLICATION_ENTRY,
            "author_pub_id": scholar_pub_id, "bib": {}, "filled": False}
        filled = self._guard(self._s.fill, pub)
        bib = filled.get("bib", {})
        authors = [a.strip() for a in (bib.get("author") or "").split(" and ") if a.strip()]
        year = bib.get("pub_year")
        return {"titre": bib.get("title"), "auteurs": authors,
                "date_publication_raw": str(year) if year else None,  # scholarly ne conserve que l'année
                "journal": bib.get("journal"), "conference": bib.get("conference"), "volume": bib.get("volume"),
                "numero": bib.get("number"), "pages": bib.get("pages"), "publisher": bib.get("publisher"),
                "abstract": bib.get("abstract"), "pdf_url": filled.get("eprint_url"),
                "external_url": filled.get("pub_url")}


def _to_year(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def make_backend(settings: dict[str, Any]) -> ScholarBackend:
    """Instancie le backend choisi dans ``settings['scraping']['backend']``."""
    cfg = settings["scraping"]
    kind = cfg.get("backend", "http")
    if kind == "http":
        return HttpScholarBackend(cfg["user_agent"], cfg["timeout"], cfg["retries"], cfg["backoff_base"],
                                  min_delay=cfg["min_delay"], max_delay=cfg["max_delay"],
                                  respect_robots=cfg.get("respect_robots_txt", True),
                                  allow_search=cfg.get("allow_author_search", False))
    if kind == "scholarly":
        if cfg.get("respect_robots_txt", True):
            raise ValueError("Le backend 'scholarly' n'est pas compatible avec respect_robots_txt: true (il utilise la recherche "
                             "d'auteurs et la pagination interdites par robots.txt). Utilisez le backend 'http', ou "
                             "désactivez explicitement respect_robots_txt en connaissance de cause.")
        return ScholarlyBackend(min_delay=cfg["min_delay"], max_delay=cfg["max_delay"])
    raise ValueError(f"Backend inconnu : {kind!r} (attendu : 'http' ou 'scholarly')")


# =========================================================================== collecteur
def _split_listing_authors(listing: Optional[str]) -> list[str]:
    if not listing:
        return []
    return [a.strip() for a in re.split(r",", listing) if a.strip() and a.strip() not in {"...", "…"}]


def new_publication_record(state: dict[str, Any], row: dict[str, Any], fetch_details: bool) -> dict[str, Any]:
    """Crée l'enregistrement brut d'une publication à partir d'une ligne de la liste du profil."""
    cid = state["chercheur_id"]
    pid = row.get("scholar_pub_id") or hashlib.sha1(f"{row.get('titre')}|{row.get('annee')}".encode()).hexdigest()[:12]
    listing = row.get("auteurs_listing")
    return {
        "raw_pub_id": f"{cid}::{pid}", "chercheur_id": cid, "scholar_id": state.get("scholar_id"),
        "scholar_pub_id": pid, "titre": row.get("titre"), "auteurs": _split_listing_authors(listing),
        "auteurs_truncated": bool(listing and is_truncated(listing)),
        "venue_raw": row.get("venue_raw"), "annee": row.get("annee"), "citations": row.get("citations"),
        "scholar_url": row.get("scholar_url"), "date_publication_raw": None, "journal": None,
        "conference": None, "volume": None, "numero": None, "pages": None, "publisher": None,
        "pdf_url": None, "external_url": None, "doi": None, "abstract": None, "abstract_source": None,
        "abstract_status": "not_found", "abstract_attempts": [],
        "detail_status": "pending" if fetch_details else "skipped",
        "scrape_status": "list_only", "scrape_error": None, "scraped_at": _now(),
    }


def merge_publication_detail(pub: dict[str, Any], detail: dict[str, Any]) -> None:
    """Fusionne la page de détail dans l'enregistrement (sans jamais écraser par une valeur vide)."""
    for key in ("date_publication_raw", "journal", "conference", "volume", "numero", "pages", "publisher",
                "pdf_url", "external_url"):
        if detail.get(key):
            pub[key] = detail[key]
    if detail.get("auteurs"):
        pub["auteurs"], pub["auteurs_truncated"] = detail["auteurs"], False
    abstract = detail.get("abstract")
    if abstract:
        pub["abstract"] = abstract
        pub["abstract_source"] = "google_scholar"
        pub["abstract_status"] = "truncated" if is_truncated(abstract) else "found"
    pub["detail_status"], pub["scrape_status"], pub["scrape_error"] = "done", "detail_ok", None


class ScholarCollector:
    """Orchestre résolution de profil → profil → publications, avec cache et reprise."""

    def __init__(self, backend: ScholarBackend, settings: dict[str, Any], raw_dir: Path, cache_dir: Path,
                 roster_names: Optional[list[str]] = None, overrides_path: Optional[Path] = None) -> None:
        self.backend = backend
        self.cfg = settings["scraping"]
        self.raw_dir = Path(raw_dir)
        self.cache_dir = Path(cache_dir) / "scholar"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.roster_names = roster_names or []
        self.overrides = self._load_overrides(Path(overrides_path)) if overrides_path else {}
        self.states: dict[str, dict[str, Any]] = {}
        for path in sorted(self.cache_dir.glob("*.json")):
            state = read_json(path)
            if state and "chercheur_id" in state:
                self.states[state["chercheur_id"]] = state
        # données de repli OpenAlex (voir src/data/openalex_source.py) : jamais prioritaires sur Scholar
        self.fallback_dir = Path(cache_dir) / "openalex"
        self.fallback_states: dict[str, dict[str, Any]] = {}
        for path in sorted(self.fallback_dir.glob("*.json")) if self.fallback_dir.exists() else []:
            state = read_json(path)
            if state and "chercheur_id" in state:
                self.fallback_states[state["chercheur_id"]] = state

    # ------------------------------------------------------------------ persistance
    @staticmethod
    def _load_overrides(path: Path) -> dict[str, str]:
        if not path.exists():
            return {}
        with path.open(encoding="utf-8", newline="") as fh:
            return {r["chercheur_id"].strip(): r["scholar_id"].strip() for r in csv.DictReader(fh)
                    if r.get("chercheur_id") and r.get("scholar_id")}

    def _cache_path(self, chercheur_id: str) -> Path:
        return self.cache_dir / f"{chercheur_id}.json"

    def save_state(self, state: dict[str, Any]) -> None:
        state["updated_at"] = _now()
        write_json(self._cache_path(state["chercheur_id"]), state)
        self.states[state["chercheur_id"]] = state

    def effective_states(self) -> dict[str, dict[str, Any]]:
        """États retenus : Scholar dès qu'il a fourni des publications, sinon le repli OpenAlex, sinon l'état Scholar."""
        out = dict(self.states)
        for cid, fallback in self.fallback_states.items():
            scholar = self.states.get(cid)
            scholar_ok = bool(scholar and scholar.get("scholar_profile_status") == "matched" and scholar.get("publications"))
            if not scholar_ok and fallback.get("scholar_profile_status") == "matched":
                out[cid] = fallback
        return out

    def save_any(self, state: dict[str, Any]) -> None:
        """Sauvegarde un état Scholar OU un état de repli OpenAlex dans son propre fichier de cache."""
        if state.get("data_source") == "openalex":
            self.fallback_dir.mkdir(parents=True, exist_ok=True)
            write_json(self.fallback_dir / f"{state['chercheur_id']}.json", state)
            self.fallback_states[state["chercheur_id"]] = state
        else:
            self.save_state(state)

    def write_consolidated(self) -> None:
        """Régénère ``scholars_raw.json`` / ``publications_raw.json`` / CSV de vérification manuelle."""
        scholars, publications = [], []
        for state in self.effective_states().values():
            scholars.append({k: v for k, v in state.items() if k != "publications"}
                            | {"n_publications_collected": len(state.get("publications", []))})
            publications.extend(state.get("publications", []))
        write_json(self.raw_dir / "scholars_raw.json", scholars)
        write_json(self.raw_dir / "publications_raw.json", publications)
        self._write_review_csv()

    def _write_review_csv(self) -> None:
        rows = []
        for state in self.states.values():
            if state.get("scholar_profile_status") in {"ambiguous", "not_found"} and state.get("candidates"):
                for cand in state["candidates"]:
                    rows.append({"chercheur_id": state["chercheur_id"], "nom_complet": state["nom_complet"],
                                 "statut": state["scholar_profile_status"], "candidat_scholar_id": cand.get("scholar_id"),
                                 "candidat_nom": cand.get("name"), "candidat_affiliation": cand.get("affiliation"),
                                 "score": cand.get("score"), "url": cand.get("scholar_url")})
        path = self.raw_dir / "scholar_candidates_review.csv"
        fields = ["chercheur_id", "nom_complet", "statut", "candidat_scholar_id", "candidat_nom",
                  "candidat_affiliation", "score", "url"]
        with path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    # ------------------------------------------------------------------ étapes
    @staticmethod
    def is_done(state: Optional[dict[str, Any]]) -> bool:
        """Un chercheur est terminé si non trouvé/ambigu, ou apparié avec collecte complète."""
        if not state:
            return False
        status = state.get("scholar_profile_status")
        return status in {"not_found", "ambiguous"} or (status == "matched" and state.get("collection_status") == "complete")

    def override_pending(self, rec: dict[str, Any], state: Optional[dict[str, Any]]) -> bool:
        """Vrai si une validation manuelle existe mais n'a pas encore été appliquée à cet état."""
        return rec["chercheur_id"] in self.overrides and (state or {}).get("match_method") != "manual_override"

    def _new_state(self, rec: dict[str, Any]) -> dict[str, Any]:
        keys = ("chercheur_id", "nom_complet", "etablissement", "laboratoire", "equipe", "type_membre")
        return {**{k: rec.get(k) for k in keys}, "scholar_profile_status": "pending", "status_detail": None,
                "profile_match_confidence": None, "match_method": None, "candidates": [], "scholar_id": None,
                "scholar_url": None, "profile": None, "coauthor_hits": [], "publications": [],
                "collection_status": "n/a", "backend": self.backend.name, "updated_at": _now()}

    def _search_and_decide(self, rec: dict[str, Any]) -> MatchDecision:
        m = self.cfg["matching"]
        override = self.overrides.get(rec["chercheur_id"])
        if override:
            return MatchDecision("matched", 1.0, override, "profil validé manuellement (scholar_overrides.csv)",
                                 [], method="manual_override")
        seen: dict[str, dict[str, Any]] = {}
        decision: Optional[MatchDecision] = None
        for template in self.cfg["search_queries"]:
            for cand in self.backend.search_authors(template.format(name=rec["nom_complet"])):
                seen.setdefault(cand["scholar_id"], cand)
            decision = decide_match(rec, list(seen.values()), accept=m["accept_threshold"],
                                    ambiguous=m["ambiguous_threshold"], min_margin=m["min_margin"],
                                    min_name_score=m["min_name_score"])
            if decision.status in {"matched", "ambiguous"}:
                break
        assert decision is not None
        return decision

    def _resolve(self, state: dict[str, Any], rec: dict[str, Any]) -> None:
        decision = self._search_and_decide(rec)
        state.update(scholar_profile_status=decision.status, profile_match_confidence=decision.confidence,
                     status_detail=decision.detail, candidates=decision.candidates, match_method=decision.method,
                     scholar_id=decision.scholar_id,
                     scholar_url=profile_url(decision.scholar_id) if decision.scholar_id else None)
        logger.info("Scholar profile %s for %s (%s)", decision.status, state["nom_complet"], decision.detail)

    def _fetch_profile(self, state: dict[str, Any], max_publications: int) -> None:
        author = self.backend.fetch_author(state["scholar_id"], max_publications)
        state["profile"] = {k: author.get(k) for k in ("name", "affiliation", "email_domain", "interests",
                                                       "metrics", "since_year", "coauthors")}
        if state["match_method"] == "auto_scoring":
            bonus, hits = coauthor_bonus(author.get("coauthors", []), self.roster_names, state["nom_complet"])
            state["coauthor_hits"] = hits
            state["profile_match_confidence"] = round(min(1.0, (state["profile_match_confidence"] or 0) + bonus), 3)
        fetch_details = self.cfg.get("fetch_publication_details", True)
        state["publications"] = [new_publication_record(state, row, fetch_details) for row in author["publications"]]
        logger.info("%d publications collected for %s", len(state["publications"]), state["nom_complet"])

    def _fetch_details(self, state: dict[str, Any]) -> None:
        pending = [p for p in state["publications"] if p["detail_status"] in {"pending", "error"}]
        for i, pub in enumerate(pending, start=1):
            try:
                detail = self.backend.fetch_publication_detail(state["scholar_id"], pub["scholar_pub_id"])
                merge_publication_detail(pub, detail)
            except BlockedError:
                raise
            except (TransientError, ScholarParseError, requests.RequestException) as exc:
                pub["detail_status"], pub["scrape_status"], pub["scrape_error"] = "error", "detail_error", str(exc)
                logger.warning("Détail indisponible pour « %s » : %s", (pub["titre"] or "")[:60], exc)
            if i % DETAIL_CHECKPOINT_EVERY == 0:
                self.save_state(state)

    def process_researcher(self, rec: dict[str, Any], max_publications: int, resume: bool = True) -> dict[str, Any]:
        """Traite un chercheur ; propage uniquement ``BlockedError`` (après sauvegarde de l'état)."""
        state = self.states.get(rec["chercheur_id"]) if resume else None
        if state is None or self.override_pending(rec, state):
            state = self._new_state(rec)
        try:
            if state["scholar_profile_status"] in {"pending", "blocked", "error"}:
                self._resolve(state, rec)
                self.save_state(state)
            if state["scholar_profile_status"] == "matched":
                if state["profile"] is None:
                    self._fetch_profile(state, max_publications)
                    self.save_state(state)
                if self.cfg.get("fetch_publication_details", True):
                    self._fetch_details(state)
                else:                                     # mode « profil seul » : les détails restent « skipped », jamais « pending »
                    for pub in state["publications"]:
                        if pub["detail_status"] == "pending":
                            pub["detail_status"] = "skipped"
                all_ok = all(p["detail_status"] in {"done", "skipped"} for p in state["publications"])
                # "partial" : une reprise retentera uniquement les détails manquants ou en erreur
                state["collection_status"] = "complete" if all_ok else "partial"
        except BlockedError as exc:
            self._mark_blocked(state, exc)
            raise
        except SearchNotAllowedError as exc:
            # ni erreur ni « non trouvé » : il manque un Scholar ID fourni à la main (ou l'opt-in explicite)
            state["scholar_profile_status"], state["status_detail"] = "pending", f"search_not_allowed: {exc}"
        except Exception as exc:  # noqa: BLE001 — une erreur sur un chercheur ne doit pas tout arrêter
            logger.error("Erreur pour %s : %s", state["nom_complet"], exc, exc_info=logger.isEnabledFor(logging.DEBUG))
            if state["scholar_profile_status"] in {"pending", "blocked", "error"}:
                state["scholar_profile_status"] = "error"
            else:
                state["collection_status"] = "error"
            state["status_detail"] = f"{type(exc).__name__}: {exc}"
        self.save_state(state)
        return state

    def _mark_blocked(self, state: dict[str, Any], exc: Exception) -> None:
        if state["scholar_profile_status"] in {"pending", "blocked", "error"}:
            state["scholar_profile_status"] = "blocked"
        else:
            state["collection_status"] = "blocked"
        state["status_detail"] = f"blocked: {exc}"
        self.save_state(state)

    # ------------------------------------------------------------------ boucle principale
    def run(self, researchers: list[dict[str, Any]], max_publications: int, resume: bool = True) -> dict[str, Any]:
        """Traite la liste ; s'arrête proprement au premier blocage (données déjà collectées conservées)."""
        summary: dict[str, Any] = {"processed": 0, "skipped_done": 0, "blocked": False, "blocked_at": None,
                                   "needs_scholar_id": 0}
        every = max(1, int(self.cfg.get("checkpoint_every", 1)))
        try:
            for i, rec in enumerate(researchers, start=1):
                state = self.states.get(rec["chercheur_id"])
                if resume and self.is_done(state) and not self.override_pending(rec, state):
                    summary["skipped_done"] += 1
                    logger.info("Déjà traité, ignoré : %s", rec["nom_complet"])
                    continue
                known_id = rec["chercheur_id"] in self.overrides or bool((state or {}).get("scholar_id"))
                if not self.backend.search_allowed and not known_id:
                    summary["needs_scholar_id"] += 1       # aucune requête possible sans Scholar ID
                    continue
                logger.info("Processing %s (%d/%d)", rec["nom_complet"], i, len(researchers))
                self.process_researcher(rec, max_publications, resume)
                summary["processed"] += 1
                if summary["processed"] % every == 0:
                    self.write_consolidated()
                    logger.info("Checkpoint saved")
        except BlockedError as exc:
            summary.update(blocked=True, blocked_at=rec["chercheur_id"])
            logger.error("BLOCAGE détecté (%s). Données sauvegardées ; relancer plus tard avec --resume. "
                         "Aucun contournement n'est tenté.", exc)
        finally:
            self.write_consolidated()
        status_counts: dict[str, int] = {}
        for rec in researchers:
            st = self.states.get(rec["chercheur_id"], {}).get("scholar_profile_status", "pending")
            status_counts[st] = status_counts.get(st, 0) + 1
        summary["profile_status_counts"] = status_counts
        return summary
