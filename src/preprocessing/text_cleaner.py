"""Nettoyage de texte : deux versions (lumière / NLP) sans détruire le sens scientifique.

Choix méthodologique : pas de suppression de stop-words ni de stemming. Les modèles
d'embeddings modernes exploitent la phrase entière ; on se limite à retirer le bruit
(HTML, caractères de contrôle, espaces multiples) et à normaliser la casse pour la
version NLP. Les symboles scientifiques (α, µ, ±, °, ², %, +, -, /) sont conservés.
"""
from __future__ import annotations

import html
import re
import unicodedata
from difflib import SequenceMatcher
from urllib.parse import unquote

_TAG_RE = re.compile(r"<[^>]+>")
_CONTROL_RE = re.compile(r"[\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f-\u009f​-‏  ‪-‮⁠﻿]")
_SPACE_RE = re.compile(r"\s+")
_ABSTRACT_LABEL_RE = re.compile(r"^\s*(abstract|résumé|resume|summary|description)\s*[:.\-–—]?\s+", re.IGNORECASE)
_TRAILING_ELLIPSIS_RE = re.compile(r"\s*(…|\.{3,})\s*$")
_LIGATURES = {"ﬁ": "fi", "ﬂ": "fl", "ﬀ": "ff", "ﬃ": "ffi", "ﬄ": "ffl"}
_SPECIAL_SPACES = dict.fromkeys(map(ord, "              　"), " ")


_AFFILIATION_RE = re.compile(
    r"\b(universit\w*|laborator\w*|facult\w*|department|d[ée]partement|institute|institut|school|college|"
    r"morocco|maroc|casablanca|france|spain|usa)\b", re.IGNORECASE)
_SENTENCE_CUES_RE = re.compile(r"\b(we|this paper|this study|this article|propose\w*|present\w*|show\w*|results?|aims?|"
                               r"investigat\w*|study|studies|approach|method)\b", re.IGNORECASE)


def looks_like_affiliation(text: str | None) -> bool:
    """Vrai si ``text`` ressemble à une liste d'affiliations d'auteurs plutôt qu'à un abstract.

    Certaines API renvoient « 1Research Laboratory, … University of Casablanca, Morocco 2Laboratory of … » dans le champ
    abstract. Critère prudent : au moins 2 mots d'affiliation, texte court, et aucune tournure d'abstract (« we »,
    « this paper », « results »…).
    """
    if not text:
        return False
    cleaned = clean_text_light(text) or ""
    hits = len(_AFFILIATION_RE.findall(cleaned))
    numbered = bool(re.match(r"^\d+\s?[A-Z]", cleaned))
    if _SENTENCE_CUES_RE.search(cleaned):
        return False
    return (hits >= 2 and len(cleaned) < 600) or (numbered and hits >= 1)


def strip_html(text: str) -> str:
    """Supprime les balises HTML/JATS et décode les entités (``&amp;`` → ``&``)."""
    if "<" in text:
        text = _TAG_RE.sub(" ", text)
    return html.unescape(text)


def clean_text_light(text: str | None) -> str | None:
    """Nettoyage léger (casse conservée) : HTML, Unicode NFC, ligatures, contrôle, espaces.

    Retourne ``None`` si le texte est vide ou absent (une absence reste une absence).
    """
    if text is None or (isinstance(text, float) and text != text):
        return None
    text = strip_html(str(text))
    text = unicodedata.normalize("NFC", text)
    for lig, repl in _LIGATURES.items():
        text = text.replace(lig, repl)
    text = text.translate(_SPECIAL_SPACES)
    text = _CONTROL_RE.sub("", text)
    text = _SPACE_RE.sub(" ", text).strip()
    return text or None


def is_truncated(text: str | None) -> bool:
    """Vrai si le texte se termine par « … » ou « ... » (aperçu tronqué par Google Scholar)."""
    return bool(text and _TRAILING_ELLIPSIS_RE.search(text.strip()))


def clean_abstract_for_nlp(text: str | None) -> str | None:
    """Version NLP d'un abstract : nettoyage léger + minuscules + sans étiquette « Abstract »
    ni points de suspension finaux de troncature.
    """
    cleaned = clean_text_light(text)
    if cleaned is None:
        return None
    cleaned = _ABSTRACT_LABEL_RE.sub("", cleaned)
    cleaned = _TRAILING_ELLIPSIS_RE.sub("", cleaned)
    cleaned = _SPACE_RE.sub(" ", cleaned).strip().lower()
    return cleaned or None


def fold_ascii(text: str) -> str:
    """Supprime les accents (« Étienne » → « Etienne »)."""
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")


def normalize_for_matching(text: str | None) -> str:
    """Forme canonique pour comparer des titres/noms : ASCII, minuscules, alphanumérique seul."""
    if not text:
        return ""
    text = fold_ascii(strip_html(str(text))).lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return _SPACE_RE.sub(" ", text).strip()


def slugify(text: str) -> str:
    """Identifiant ASCII ``a_b_c`` à partir d'un texte quelconque."""
    return normalize_for_matching(text).replace(" ", "_")


_DOI_PREFIX_RE = re.compile(r"^(https?://(dx\.)?doi\.org/|doi:\s*)", re.IGNORECASE)
_DOI_SEARCH_RE = re.compile(r"10\.\d{4,9}/[^\s\"'<>?#&]+", re.IGNORECASE)  # « & » = séparateur de paramètres d'URL
_DOI_FULL_RE = re.compile(r"^10\.\d{4,9}/\S+$")


def normalize_doi(value: str | None) -> str | None:
    """Forme canonique d'un DOI (minuscules, sans préfixe URL) ou ``None`` s'il est invalide."""
    if not value:
        return None
    doi = _DOI_PREFIX_RE.sub("", unquote(str(value)).strip()).strip().lower().rstrip(".,;")
    return doi if _DOI_FULL_RE.match(doi) else None


def extract_doi(text: str | None) -> str | None:
    """Repère un DOI dans une URL ou un texte libre (ex. lien éditeur contenant ``/10.1016/...``)."""
    if not text:
        return None
    match = _DOI_SEARCH_RE.search(unquote(str(text)))
    return normalize_doi(match.group(0)) if match else None


def title_similarity(a: str | None, b: str | None) -> float:
    """Similarité ∈ [0, 1] entre deux titres (forme normalisée, ratio de SequenceMatcher)."""
    na, nb = normalize_for_matching(a), normalize_for_matching(b)
    if not na or not nb:
        return 0.0
    return SequenceMatcher(None, na, nb).ratio()


def build_embedding_text(title: str | None, abstract_clean: str | None, min_abstract_chars: int = 50) -> tuple[str | None, str]:
    """Construit le texte à encoder et sa provenance.

    Returns:
        ``(texte, embedding_source)`` avec ``embedding_source`` dans
        ``{"title_abstract", "title_only", "abstract_only", "none"}``.
        Un abstract plus court que ``min_abstract_chars`` est ignoré (bruit).
    """
    has_abstract = bool(abstract_clean) and len(abstract_clean) >= min_abstract_chars
    if title and has_abstract:
        sep = " " if title.rstrip()[-1:] in ".!?:;" else ". "
        return f"{title.rstrip()}{sep}{abstract_clean}", "title_abstract"
    if title:
        return title, "title_only"
    if has_abstract:
        return abstract_clean, "abstract_only"
    return None, "none"
