"""Lecture HORS LIGNE de pages de profil Google Scholar enregistrées à la main (Ctrl+S dans le navigateur).

Principe : la personne ouvre elle-même les profils dans son navigateur (comme n'importe quel visiteur, en
résolvant elle-même un éventuel contrôle « not a robot ») et enregistre la page ; ce module lit ensuite les
fichiers HTML locaux. Aucune requête automatique n'est envoyée à Google : il ne s'agit pas de scraping mais
d'une lecture de fichiers, et le contrôle anti-robot n'est jamais contourné.
"""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Any, Optional

from src.data.scholar_parsing import ScholarParseError, detect_block, parse_profile, profile_url
from src.data.scholar_scraper import new_publication_record, _now
from src.utils.retry import BlockedError

_CITATION_ID_RE = re.compile(r"citation_for_view=([\w-]+):")
_USER_ID_RE = re.compile(r"[?&;]user=([\w-]{8,})")


def read_html(path: Path) -> str:
    """Lit un fichier HTML enregistré (UTF-8 ; caractères invalides remplacés plutôt que d'échouer)."""
    return Path(path).read_bytes().decode("utf-8", errors="replace")


def extract_scholar_id(html: str) -> Optional[str]:
    """Identifiant Scholar du profil : celui des liens de publications, à défaut le plus fréquent ``user=``."""
    ids = _CITATION_ID_RE.findall(html)
    if ids:
        return Counter(ids).most_common(1)[0][0]
    users = _USER_ID_RE.findall(html)
    return Counter(users).most_common(1)[0][0] if users else None


def inspect_saved_page(html: str) -> dict[str, Any]:
    """Diagnostic d'une page enregistrée : ``status`` ∈ {ok, robot_check, not_a_profile} + contenu analysé si ok."""
    try:
        detect_block(200, "", html)
    except BlockedError:
        return {"status": "robot_check", "detail": "page de contrôle « not a robot » enregistrée : rouvrez le profil, "
                                                   "validez le contrôle, attendez l'affichage des publications puis réenregistrez"}
    try:
        profile = parse_profile(html)
    except ScholarParseError:
        return {"status": "not_a_profile", "detail": "ce fichier n'est pas une page de profil Scholar"}
    return {"status": "ok", "scholar_id": extract_scholar_id(html), "profile": profile}


def state_from_saved_page(new_state: dict[str, Any], scholar_id: str, profile: dict[str, Any], max_publications: int,
                          source_file: str) -> dict[str, Any]:
    """Complète un état de chercheur (forme ``ScholarCollector._new_state``) avec le contenu d'une page enregistrée."""
    new_state.update(
        scholar_profile_status="matched", match_method="manual_saved_html", profile_match_confidence=1.0,
        scholar_id=scholar_id, scholar_url=profile_url(scholar_id), collection_status="complete",
        status_detail="page de profil Google Scholar enregistrée manuellement, lue hors ligne", data_source="google_scholar",
        acquisition={"method": "saved_html", "file": source_file, "read_at": _now()},
        profile={k: profile.get(k) for k in ("name", "affiliation", "email_domain", "interests", "metrics", "since_year", "coauthors")})
    rows = profile["publications"][:max_publications]
    new_state["publications"] = [new_publication_record(new_state, row, fetch_details=False) for row in rows]
    return new_state
