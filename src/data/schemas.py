"""Schémas Pydantic du projet.

Règle d'or : une valeur inconnue est ``None`` (jamais 0, jamais chaîne vide) et un champ
``*_status`` explique pourquoi elle est absente.
"""
from __future__ import annotations

import re
from datetime import date
from typing import Iterable, Literal, Optional
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, field_validator

YEAR_MIN = 1900
DOI_RE = re.compile(r"^10\.\d{4,9}/\S+$")

ScholarStatus = Literal["matched", "ambiguous", "not_found", "blocked", "error", "pending"]
AbstractStatus = Literal["found", "truncated", "too_short", "not_found", "api_error", "publisher_unavailable", "pending"]
DatePrecision = Literal["day", "month", "year"]
EmbeddingSource = Literal["title_abstract", "title_only", "abstract_only", "none"]


def year_max() -> int:
    """Année maximale plausible (année courante + 1 pour les articles « à paraître »)."""
    return date.today().year + 1


def _validate_url(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"URL invalide : {value!r}")
    return value


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class ResearcherSource(_Base):
    """Ligne du PDF « Membres FSBM » après extraction et normalisation."""

    chercheur_id: str = Field(pattern=r"^[a-z0-9]+(_[a-z0-9]+)+$")
    chercheur_source_name: str
    nom_complet: str = Field(min_length=1)
    etablissement: Optional[str] = None
    laboratoire: Optional[str] = None
    equipe: Optional[str] = None
    type_membre: Optional[str] = None


class ScholarMetrics(_Base):
    """Métriques Google Scholar. ``None`` = inconnu (≠ 0)."""

    citations_totales: Optional[int] = Field(default=None, ge=0)
    h_index: Optional[int] = Field(default=None, ge=0)
    i10_index: Optional[int] = Field(default=None, ge=0)
    # Scholar affiche « Depuis <année> » (année courante - 5). Les champs *_since_2021 ne sont
    # renseignés que si cette fenêtre vaut exactement 2021 ; sinon ils restent None.
    since_year: Optional[int] = Field(default=None, ge=1990)
    citations_since: Optional[int] = Field(default=None, ge=0)
    h_index_since: Optional[int] = Field(default=None, ge=0)
    i10_index_since: Optional[int] = Field(default=None, ge=0)
    citations_since_2021: Optional[int] = Field(default=None, ge=0)
    h_index_since_2021: Optional[int] = Field(default=None, ge=0)
    i10_index_since_2021: Optional[int] = Field(default=None, ge=0)


class ResearcherProfile(_Base):
    """Chercheur enrichi avec son profil Google Scholar (ou l'absence de profil)."""

    chercheur_id: str
    nom_complet: str
    scholar_id: Optional[str] = None
    scholar_url: Optional[str] = None
    affiliation: Optional[str] = None
    email_domain: Optional[str] = None
    interests: list[str] = Field(default_factory=list)
    laboratoire: Optional[str] = None
    equipe: Optional[str] = None
    metriques: ScholarMetrics = Field(default_factory=ScholarMetrics)
    scholar_profile_status: ScholarStatus = "pending"
    profile_match_confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    status_detail: Optional[str] = None

    _url = field_validator("scholar_url")(_validate_url)


class Publication(_Base):
    """Publication dédupliquée (une ligne par article, tous chercheurs confondus)."""

    article_id: str = Field(pattern=r"^art_\d+$")
    titre: str = Field(min_length=1)
    auteurs: list[str] = Field(default_factory=list)
    annee: Optional[int] = None
    date_publication: Optional[str] = None
    date_precision: Optional[DatePrecision] = None
    journal: Optional[str] = None
    conference: Optional[str] = None
    volume: Optional[str] = None
    numero: Optional[str] = None
    pages: Optional[str] = None
    publisher: Optional[str] = None
    citations: Optional[int] = Field(default=None, ge=0)
    scholar_url: Optional[str] = None
    pdf_url: Optional[str] = None
    doi: Optional[str] = None
    abstract: Optional[str] = None
    abstract_clean: Optional[str] = None
    abstract_source: Optional[str] = None
    abstract_status: AbstractStatus = "not_found"
    embedding_source: EmbeddingSource = "none"
    scrape_status: Optional[str] = None

    _urls = field_validator("scholar_url", "pdf_url")(_validate_url)

    @field_validator("auteurs")
    @classmethod
    def _authors_are_clean(cls, value: list[str]) -> list[str]:
        if any((not isinstance(a, str)) or not a.strip() for a in value):
            raise ValueError("liste d'auteurs invalide (élément vide ou non textuel)")
        return value

    @field_validator("annee")
    @classmethod
    def _plausible_year(cls, value: Optional[int]) -> Optional[int]:
        if value is not None and not (YEAR_MIN <= value <= year_max()):
            raise ValueError(f"année implausible : {value}")
        return value

    @field_validator("doi")
    @classmethod
    def _doi_format(cls, value: Optional[str]) -> Optional[str]:
        if value is not None and not DOI_RE.match(value):
            raise ValueError(f"DOI invalide : {value!r}")
        return value


class ResearcherPublicationLink(_Base):
    """Lien N-N chercheur ↔ publication."""

    chercheur_id: str
    article_id: str
    raw_pub_id: Optional[str] = None


def find_duplicates(values: Iterable[str]) -> list[str]:
    """Retourne les valeurs présentes plus d'une fois (pour contrôler l'unicité des identifiants)."""
    seen: set[str] = set()
    dup: list[str] = []
    for value in values:
        if value in seen and value not in dup:
            dup.append(value)
        seen.add(value)
    return dup


def assert_unique(values: Iterable[str], label: str) -> None:
    """Lève ``ValueError`` si des identifiants ne sont pas uniques."""
    dup = find_duplicates(values)
    if dup:
        raise ValueError(f"{label} non uniques : {dup[:5]}{'…' if len(dup) > 5 else ''}")
