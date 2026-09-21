"""Tests : schémas Pydantic et règles de validation (métriques ≥ 0, année plausible, URL, auteurs, unicité)."""
from datetime import date

import pytest
from pydantic import ValidationError

from src.data.schemas import (Publication, ResearcherProfile, ResearcherSource, ScholarMetrics, assert_unique,
                              find_duplicates)
from src.preprocessing.data_cleaner import validate_articles


def make_pub(**overrides):
    base = {"article_id": "art_00001", "titre": "A title", "auteurs": ["Jane Doe", "John Roe"], "annee": 2021,
            "citations": 3, "doi": "10.1000/xyz", "scholar_url": "https://scholar.google.com/x"}
    return {**base, **overrides}


def test_valid_publication_and_missing_values_are_none_not_zero():
    pub = Publication(**make_pub(citations=None, doi=None, annee=None))
    assert pub.citations is None and pub.doi is None and pub.annee is None
    assert pub.abstract_status == "not_found" and pub.embedding_source == "none"


@pytest.mark.parametrize("overrides", [
    {"citations": -1},
    {"annee": 1200},
    {"annee": date.today().year + 5},
    {"scholar_url": "ftp://x.org/file"},
    {"pdf_url": "not a url"},
    {"auteurs": ["Jane", ""]},
    {"doi": "11.1/bad"},
    {"article_id": "article-1"},
    {"abstract_status": "invented"},
    {"unknown_field": 1},
])
def test_invalid_publication_rejected(overrides):
    with pytest.raises(ValidationError):
        Publication(**make_pub(**overrides))


def test_metrics_must_be_non_negative_and_default_to_none():
    assert ScholarMetrics().h_index is None
    assert ScholarMetrics(h_index=0).h_index == 0
    for field in ("citations_totales", "h_index", "i10_index", "citations_since_2021"):
        with pytest.raises(ValidationError):
            ScholarMetrics(**{field: -1})


def test_researcher_profile_confidence_range_and_status():
    ok = ResearcherProfile(chercheur_id="fsbm_a_b", nom_complet="A B", scholar_profile_status="matched",
                           profile_match_confidence=0.9, scholar_url="https://scholar.google.com/citations?user=abc")
    assert ok.profile_match_confidence == 0.9
    with pytest.raises(ValidationError):
        ResearcherProfile(chercheur_id="x", nom_complet="A", profile_match_confidence=1.5)
    with pytest.raises(ValidationError):
        ResearcherProfile(chercheur_id="x", nom_complet="A", scholar_profile_status="maybe")
    with pytest.raises(ValidationError):
        ResearcherProfile(chercheur_id="x", nom_complet="A", scholar_url="javascript:void(0)")


def test_researcher_source_id_format():
    ResearcherSource(chercheur_id="fsbm_driss_bouggar", chercheur_source_name="DRISS.BOUGGAR", nom_complet="Driss Bouggar")
    with pytest.raises(ValidationError):
        ResearcherSource(chercheur_id="Driss Bouggar", chercheur_source_name="x", nom_complet="x")


def test_unique_identifier_helpers():
    assert find_duplicates(["a", "b", "a", "c", "a", "b"]) == ["a", "b"]
    assert_unique(["a", "b"], "ids")
    with pytest.raises(ValueError, match="non uniques"):
        assert_unique(["a", "a"], "ids")


def test_validate_articles_reports_field_level_errors():
    good = make_pub()
    bad = make_pub(article_id="art_00002", citations=-5)
    errors = validate_articles([good, bad])
    assert len(errors) == 1 and errors[0]["article_id"] == "art_00002" and errors[0]["field"] == "citations"
