"""Parsing HTML de Google Scholar (fonctions pures, testables sans réseau).

Google Scholar peut modifier son HTML à tout moment : les sélecteurs sont regroupés ici,
et une structure inattendue lève ``ScholarParseError`` (jamais de données inventées).
"""
from __future__ import annotations

import re
from typing import Any, Optional
from urllib.parse import parse_qs, urljoin, urlparse

from bs4 import BeautifulSoup

from src.utils.retry import BlockedError

BASE_URL = "https://scholar.google.com"
_BLOCK_MARKERS = ("unusual traffic", "not a robot", "gs_captcha", "recaptcha", "/sorry/",
                  "system can't perform the operation now", "system can&#39;t perform the operation now")


class ScholarParseError(RuntimeError):
    """Le HTML reçu ne correspond pas à la structure attendue (changement de mise en page ?)."""


def detect_block(status_code: int, url: str, text: str) -> None:
    """Lève ``BlockedError`` si la réponse indique un blocage / CAPTCHA (jamais contourné)."""
    if status_code in (429, 403):
        raise BlockedError(f"HTTP {status_code} — accès limité par Google Scholar")
    lowered = text.lower()          # page entière : le message de CAPTCHA arrive après ~20 000 caractères de CSS/JS
    if "/sorry/" in url or any(marker in lowered for marker in _BLOCK_MARKERS):
        raise BlockedError("Page de vérification (CAPTCHA / trafic inhabituel) détectée")


def _soup(html: str) -> BeautifulSoup:
    return BeautifulSoup(html, "html.parser")


def _text(node: Any) -> Optional[str]:
    if node is None:
        return None
    value = node.get_text(" ", strip=True)
    return value or None


def _to_int(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    digits = re.sub(r"[^\d]", "", value)
    return int(digits) if digits else None


def _email_domain(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    match = re.search(r"Verified email at\s+([\w.\-]+)", text, flags=re.IGNORECASE)
    return match.group(1).lower() if match else None


def _query_param(href: str, name: str) -> Optional[str]:
    values = parse_qs(urlparse(href).query).get(name)
    return values[0] if values else None


def profile_url(scholar_id: str) -> str:
    return f"{BASE_URL}/citations?user={scholar_id}&hl=en"


# --------------------------------------------------------------------------- recherche d'auteurs
def parse_search_results(html: str) -> list[dict[str, Any]]:
    """Candidats d'une recherche d'auteurs (``view_op=search_authors``)."""
    candidates: list[dict[str, Any]] = []
    for block in _soup(html).select("div.gsc_1usr"):
        link = block.select_one("h3.gs_ai_name a")
        if link is None:
            continue
        scholar_id = _query_param(link.get("href", ""), "user")
        if not scholar_id:
            continue
        candidates.append({
            "scholar_id": scholar_id,
            "name": _text(link),
            "affiliation": _text(block.select_one("div.gs_ai_aff")),
            "email_domain": _email_domain(_text(block.select_one("div.gs_ai_eml"))),
            "interests": [_text(a) for a in block.select("a.gs_ai_one_int") if _text(a)],
            "cited_by": _to_int(_text(block.select_one("div.gs_ai_cby"))),
            "scholar_url": profile_url(scholar_id),
        })
    return candidates


# --------------------------------------------------------------------------- profil
def parse_metrics(soup: BeautifulSoup) -> tuple[dict[str, Optional[int]], Optional[int]]:
    """Table #gsc_rsb_st : citations / h-index / i10-index (Toutes | Depuis <année>)."""
    table = soup.select_one("#gsc_rsb_st")
    metrics: dict[str, Optional[int]] = {}
    since_year: Optional[int] = None
    if table is None:
        return metrics, since_year
    header = _text(table.select_one("thead")) or ""
    year_match = re.search(r"Since\s+(\d{4})", header)
    since_year = int(year_match.group(1)) if year_match else None
    keys = {"citations": "citations", "h-index": "h_index", "i10-index": "i10_index"}
    for row in table.select("tbody tr"):
        label = (_text(row.select_one("td.gsc_rsb_sc1")) or "").strip().lower()
        values = [_to_int(_text(td)) for td in row.select("td.gsc_rsb_std")]
        key = keys.get(label)
        if key and values:
            metrics[key] = values[0]
            metrics[f"{key}_since"] = values[1] if len(values) > 1 else None
    return metrics, since_year


def parse_publication_rows(soup: BeautifulSoup) -> list[dict[str, Any]]:
    """Lignes de publications (``tr.gsc_a_tr``) de la page de profil."""
    rows: list[dict[str, Any]] = []
    for tr in soup.select("tr.gsc_a_tr"):
        link = tr.select_one("a.gsc_a_at")
        if link is None:
            continue
        href = link.get("href", "")
        grays = tr.select("div.gs_gray")
        year = _to_int(_text(tr.select_one("td.gsc_a_y span")))
        cites_cell = tr.select_one("td.gsc_a_c")
        cites_txt = _text(cites_cell)
        rows.append({
            "scholar_pub_id": _query_param(href, "citation_for_view"),
            "titre": _text(link),
            "auteurs_listing": _text(grays[0]) if grays else None,
            "venue_raw": _text(grays[1]) if len(grays) > 1 else None,
            "annee": year,
            # Scholar affiche une cellule vide pour 0 citation ; "*" = valeur non fournie.
            "citations": None if cites_txt == "*" else (_to_int(cites_txt) or 0),
            "scholar_url": urljoin(BASE_URL, href) if href else None,
        })
    return rows


def parse_profile(html: str) -> dict[str, Any]:
    """Page de profil : identité, domaines, métriques, co-auteurs, publications, pagination."""
    soup = _soup(html)
    name = _text(soup.select_one("#gsc_prf_in"))
    if name is None:
        raise ScholarParseError("Élément #gsc_prf_in introuvable : profil inexistant ou HTML modifié")
    info_lines = soup.select("#gsc_prf_i .gsc_prf_il")
    affiliation = _text(info_lines[0]) if info_lines else None
    if affiliation and affiliation.lower().startswith("verified email"):
        affiliation = None
    metrics, since_year = parse_metrics(soup)
    more_btn = soup.select_one("#gsc_bpf_more")
    return {
        "name": name,
        "affiliation": affiliation,
        "email_domain": _email_domain(_text(soup.select_one("#gsc_prf_ivh"))),
        "interests": [_text(a) for a in soup.select("#gsc_prf_int a") if _text(a)],
        "metrics": metrics,
        "since_year": since_year,
        "coauthors": [_text(a) for a in soup.select("#gsc_rsb_co a[href*='user=']") if _text(a)],
        "publications": parse_publication_rows(soup),
        "has_more": bool(more_btn is not None and more_btn.get("disabled") is None),
    }


# --------------------------------------------------------------------------- détail d'une publication
_DETAIL_FIELDS = {
    "authors": "auteurs_raw", "publication date": "date_publication_raw", "journal": "journal",
    "conference": "conference", "volume": "volume", "issue": "numero", "pages": "pages",
    "publisher": "publisher", "description": "abstract", "book": "venue_other", "source": "venue_other",
}


def parse_publication_detail(html: str) -> dict[str, Any]:
    """Page ``view_citation`` : métadonnées complètes et « Description » (abstract, parfois tronqué)."""
    soup = _soup(html)
    title_node = soup.select_one("#gsc_oci_title")
    if title_node is None:
        raise ScholarParseError("Élément #gsc_oci_title introuvable : HTML modifié ?")
    detail: dict[str, Any] = {"titre": _text(title_node)}
    external = soup.select_one("a.gsc_oci_title_link") or title_node.select_one("a")
    detail["external_url"] = external.get("href") if external is not None else None
    pdf = soup.select_one("#gsc_oci_title_gg a") or soup.select_one(".gsc_vcd_title_ggi a")
    detail["pdf_url"] = pdf.get("href") if pdf is not None else None
    for block in soup.select(".gs_scl"):
        label = (_text(block.select_one(".gsc_oci_field")) or "").lower()
        target = _DETAIL_FIELDS.get(label)
        value_node = block.select_one(".gsc_oci_value")
        if target is None or value_node is None or target in detail:
            continue
        if target == "abstract":
            parts = [_text(p) for p in value_node.select(".gsh_csp")]
            value = " ".join(p for p in parts if p) or _text(value_node)
        else:
            value = _text(value_node)
        if value:
            detail[target] = value
    raw_authors = detail.pop("auteurs_raw", None)
    detail["auteurs"] = [a.strip() for a in raw_authors.split(",") if a.strip()] if raw_authors else []
    return detail
