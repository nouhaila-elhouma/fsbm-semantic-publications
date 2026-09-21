"""Tests : repli OpenAlex (sélection prudente des profils, fusion des fragments, étiquetage, priorité de Scholar)."""
import copy

import pytest

from src.data.openalex_source import HASSAN_II_INSTITUTION, OpenAlexFallback, OpenAlexSource, _short_id
from src.data.scholar_scraper import ScholarBackend, ScholarCollector
from src.utils.config import load_settings
from src.utils.io import read_json


def author(oid, name, works, inst=HASSAN_II_INSTITUTION, h=5, cites=100):
    inst_obj = [{"id": f"https://openalex.org/{inst}", "display_name": "University of Hassan II Casablanca"}] if inst else []
    return {"id": f"https://openalex.org/{oid}", "display_name": name, "works_count": works, "cited_by_count": cites,
            "summary_stats": {"h_index": h, "i10_index": 2}, "last_known_institutions": inst_obj, "affiliations": [],
            "topics": [{"display_name": "Topic Modeling"}, {"display_name": "Deep learning"}]}


def work(wid, title, year=2021, doi="https://doi.org/10.1000/AbC", abstract=None, source="Procedia CS"):
    inverted = {w: [i] for i, w in enumerate(abstract.split())} if abstract else None
    return {"id": f"https://openalex.org/{wid}", "display_name": title, "publication_year": year, "publication_date": f"{year}-01-01",
            "doi": doi, "cited_by_count": 7, "abstract_inverted_index": inverted, "type": "article",
            "authorships": [{"author": {"display_name": "A One"}}, {"author": {"display_name": "B Two"}}],
            "primary_location": {"source": {"display_name": source, "type": "journal", "host_organization_name": "Elsevier"}},
            "best_oa_location": {"pdf_url": "https://example.org/x.pdf"}}


class FakeOA(OpenAlexSource):
    """OpenAlexSource sans réseau : résultats programmés."""

    def __init__(self, authors, works, **kw):
        super().__init__(delay_fn=lambda: None, **kw)
        self._authors, self._works, self.requests = authors, works, []

    def search_authors(self, name, per_page=15):
        self.requests.append(("authors", name))
        return self._authors

    def fetch_works(self, author_ids, max_publications):
        self.requests.append(("works", author_ids))
        return self._works[:max_publications]


REC = {"chercheur_id": "fsbm_habib_benlahmar", "nom_complet": "Elhabib Benlahmar", "etablissement": "FSBM", "laboratoire": "L", "equipe": "E"}


def test_matched_requires_near_identical_name_and_hassan_ii_and_merges_fragments():
    src = FakeOA([author("A1", "El Habib Benlahmar", 130), author("A2", "Ben Lahmar El Habib", 2), author("A3", "Someone Else", 300),
                  author("A4", "El Habib Benlahmar", 50, inst="I999")], [work("W1", "Paper one")])
    state = src.collect(REC, ["Elhabib Benlahmar", "Habib BEN LAHMAR"], "upOdTrEAAAAJ")
    assert state["scholar_profile_status"] == "matched" and state["data_source"] == "openalex"
    assert state["openalex_ids"] == ["A1", "A2"]                       # principal + fragment ; A3 (autre nom) et A4 (autre institution) exclus
    assert src.requests[-1] == ("works", ["A1", "A2"])
    assert state["scholar_id"] == "upOdTrEAAAAJ" and state["profile"]["metrics"]["h_index"] == 5


def test_no_profile_or_wrong_institution_is_not_found_and_never_guessed():
    src = FakeOA([author("A1", "El Habib Benlahmar", 130, inst="I999"), author("A2", "Karim Tazi", 40)], [])
    state = src.collect(REC, ["Elhabib Benlahmar"], None)
    assert state["scholar_profile_status"] == "not_found" and state["publications"] == [] and state["profile"] is None


def test_two_large_homonyms_are_ambiguous():
    src = FakeOA([author("A1", "Mohamed Idiri", 80), author("A2", "Mohamed Idiri", 60)], [])
    assert src.collect({**REC, "nom_complet": "Mohamed Idiri"}, ["Mohamed Idiri"], None)["scholar_profile_status"] == "ambiguous"


def test_work_record_is_labelled_and_never_invents_values():
    src = FakeOA([], [])
    full = src.work_to_record("fsbm_x_y", "SID", work("W9", "A title", abstract="We study deep learning for medical imaging today"))
    assert full["data_source"] == "openalex" and full["scrape_status"] == "openalex" and full["abstract_source"] == "openalex"
    assert full["doi"] == "10.1000/abc" and full["journal"] == "Procedia CS" and full["publisher"] == "Elsevier"
    assert full["abstract_status"] == "found" and full["auteurs"] == ["A One", "B Two"] and full["date_publication_raw"] == "2021-01-01"
    assert full["raw_pub_id"] == "fsbm_x_y::oa:W9" and full["citations"] == 7
    empty = src.work_to_record("fsbm_x_y", None, work("W10", "No abstract", abstract=None))
    assert empty["abstract"] is None and empty["abstract_status"] == "not_found" and empty["abstract_source"] is None
    aff = src.work_to_record("fsbm_x_y", None, work("W11", "T", abstract="1Laboratory of Physics Faculty Hassan II University Casablanca"))
    assert aff["abstract"] is None                                     # liste d'affiliations ≠ abstract


def test_conference_venues_go_to_the_conference_field():
    w = work("W3", "Talk")
    w["primary_location"]["source"]["type"] = "conference"
    rec = FakeOA([], []).work_to_record("c", None, w)
    assert rec["conference"] == "Procedia CS" and rec["journal"] is None


def test_fallback_caches_per_researcher_and_resumes(tmp_path):
    src = FakeOA([author("A1", "El Habib Benlahmar", 10)], [work("W1", "P")])
    fb = OpenAlexFallback(src, tmp_path)
    counts = fb.run([REC], {}, {REC["chercheur_id"]: "SID"})
    assert counts["matched"] == 1 and (tmp_path / "openalex" / "fsbm_habib_benlahmar.json").exists()
    n = len(src.requests)
    assert OpenAlexFallback(src, tmp_path).run([REC], {}, {})["skipped_done"] == 1 and len(src.requests) == n   # aucune requête refaite


def test_api_errors_are_recorded_and_retried_on_resume(tmp_path):
    class Failing(FakeOA):
        def search_authors(self, name, per_page=15):
            raise RuntimeError("HTTP 429")

    fb = OpenAlexFallback(Failing([], []), tmp_path)
    assert fb.run([REC], {}, {})["error"] == 1
    ok = OpenAlexFallback(FakeOA([author("A1", "El Habib Benlahmar", 10)], [work("W1", "P")]), tmp_path)
    assert ok.run([REC], {}, {}, resume=True)["matched"] == 1          # les erreurs sont retentées


class _NoBackend(ScholarBackend):
    name = "fake"

    def search_authors(self, query):
        return []

    def fetch_author(self, scholar_id, max_publications):
        raise AssertionError("pas de requête Scholar dans ce test")

    def fetch_publication_detail(self, scholar_id, scholar_pub_id):
        raise AssertionError


def test_scholar_data_always_wins_over_the_openalex_fallback(tmp_path):
    cfg = load_settings()
    fb_state = {"chercheur_id": "c1", "nom_complet": "C One", "scholar_profile_status": "matched", "data_source": "openalex",
                "publications": [{"raw_pub_id": "c1::oa:W1", "chercheur_id": "c1", "titre": "From OpenAlex"}], "profile": {}}
    (tmp_path / "cache" / "openalex").mkdir(parents=True)
    (tmp_path / "cache" / "openalex" / "c1.json").write_text(__import__("json").dumps(fb_state), encoding="utf-8")
    collector = ScholarCollector(_NoBackend(), cfg, tmp_path / "raw", tmp_path / "cache")
    assert collector.effective_states()["c1"]["data_source"] == "openalex"           # Scholar bloqué / absent : repli utilisé

    collector.states["c1"] = {"chercheur_id": "c1", "nom_complet": "C One", "scholar_profile_status": "matched",
                              "publications": [{"raw_pub_id": "c1::p1", "chercheur_id": "c1", "titre": "From Scholar"}], "profile": {}}
    assert "data_source" not in collector.effective_states()["c1"]                    # Scholar a des publications : il prend la priorité
    collector.write_consolidated()
    pubs = read_json(tmp_path / "raw" / "publications_raw.json")
    assert [p["titre"] for p in pubs] == ["From Scholar"]


def test_short_id():
    assert _short_id("https://openalex.org/A5023525580") == "A5023525580"
