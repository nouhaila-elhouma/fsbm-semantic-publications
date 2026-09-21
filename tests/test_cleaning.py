"""Tests : normalisation de noms, nettoyage de texte, IDs, dates, DOI, déduplication, extraction PDF (lignes)."""
import pytest

from src.data.extract_fsbm_members import (ExtractionReport, build_members_dataframe, detect_header_mapping,
                                           make_researcher_id, name_key, normalize_person_name, rows_to_records,
                                           summarize_members, unique_researchers)
from src.preprocessing.data_cleaner import (clean_publication_record, deduplicate_publications, find_duplicate_groups,
                                            make_article_id, parse_publication_date, sanitize_count, sanitize_url)
from src.preprocessing.text_cleaner import (build_embedding_text, clean_abstract_for_nlp, clean_text_light,
                                            extract_doi, is_truncated, normalize_doi)


# ---------------------------------------------------------------- noms & identifiants
@pytest.mark.parametrize("raw,expected", [
    ("DRISS.BOUGGAR", "Driss Bouggar"),
    ("  driss   bouggar ", "Driss Bouggar"),
    ("Pr. ALI.ALAMI", "Ali Alami"),
    ("BEN M'SIK Sara", "Ben M'Sik Sara"),
    ("EL-AMRANI.Yassine", "El-Amrani Yassine"),
    ("Élise.Fontaine", "Élise Fontaine"),
    (None, ""),
])
def test_normalize_person_name(raw, expected):
    assert normalize_person_name(raw) == expected


def test_name_key_is_order_and_accent_insensitive():
    assert name_key("Driss Bouggar") == name_key("BOUGGAR Driss") == name_key("Bouggar Dríss")


def test_make_researcher_id_is_stable_and_ascii():
    assert make_researcher_id("Driss Bouggar", "FSBM") == "fsbm_driss_bouggar"
    assert make_researcher_id("Élise Ben M'Sik", "FSBM") == "fsbm_elise_ben_m_sik"
    assert make_researcher_id("Driss Bouggar", "FSBM") == make_researcher_id("Driss Bouggar", "FSBM")
    assert make_researcher_id("Sara Alami", "ENCG").startswith("encg_")


def test_make_article_id():
    assert make_article_id(1) == "art_00001"
    assert make_article_id(123) == "art_00123"


# ---------------------------------------------------------------- nettoyage de texte
def test_clean_text_light_removes_noise_but_keeps_case_and_symbols():
    raw = "  <p>Effect of  α-particles &amp; ±5 °C</p>\n\n  on  ﬁlms​ "
    assert clean_text_light(raw) == "Effect of α-particles & ±5 °C on films"


def test_clean_text_light_keeps_absence_as_none():
    assert clean_text_light(None) is None
    assert clean_text_light("   ") is None
    assert clean_text_light(float("nan")) is None


def test_clean_abstract_for_nlp():
    raw = "<jats:p>Abstract: We STUDY   CO2 capture at 25 °C   in Zeolites…</jats:p>"
    assert clean_abstract_for_nlp(raw) == "we study co2 capture at 25 °c in zeolites"


def test_clean_abstract_keeps_stop_words():
    assert "the" in clean_abstract_for_nlp("The effect of the model on the results") .split()


def test_is_truncated():
    assert is_truncated("some text …") and is_truncated("some text...")
    assert not is_truncated("complete sentence.")


def test_build_embedding_text_sources():
    text, src = build_embedding_text("A title", "x" * 60)
    assert src == "title_abstract" and text.startswith("A title. ")
    assert build_embedding_text("A title", None) == ("A title", "title_only")
    assert build_embedding_text("A title", "too short")[1] == "title_only"
    assert build_embedding_text("Why?", "y" * 60)[0].startswith("Why? y")
    assert build_embedding_text(None, None) == (None, "none")


# ---------------------------------------------------------------- dates, DOI, nombres
@pytest.mark.parametrize("raw,fallback,expected", [
    ("2021/1/1", None, ("2021-01-01", 2021, "day")),
    ("2021/03", None, ("2021-03", 2021, "month")),
    ("2019", None, ("2019", 2019, "year")),
    ("Jan 2020", None, ("2020-01", 2020, "month")),
    ("March 5, 2018", None, ("2018-03-05", 2018, "day")),
    ("5 March 2018", None, ("2018-03-05", 2018, "day")),
    (None, 2017, ("2017", 2017, "year")),
    (None, None, (None, None, None)),
    ("garbage", None, (None, None, None)),
    ("2021/13/45", None, ("2021", 2021, "year")),
])
def test_parse_publication_date(raw, fallback, expected):
    assert parse_publication_date(raw, fallback) == expected


def test_doi_normalisation_and_extraction():
    assert normalize_doi("https://doi.org/10.1016/J.Procs.2021.07.045") == "10.1016/j.procs.2021.07.045"
    assert normalize_doi("doi:10.1000/abc.") == "10.1000/abc"
    assert normalize_doi("not a doi") is None
    assert extract_doi("https://www.sciencedirect.com/x?doi=10.1016/j.x.2021.01.001&foo=1") == "10.1016/j.x.2021.01.001"
    assert extract_doi("https://example.org/page") is None


def test_sanitize_helpers_never_turn_unknown_into_zero():
    assert sanitize_count(None) is None and sanitize_count("abc") is None and sanitize_count(-3) is None
    assert sanitize_count(0) == 0 and sanitize_count("12") == 12
    assert sanitize_url("javascript:alert(1)") is None and sanitize_url("https://a.org/x") == "https://a.org/x"


def _raw(titre, chercheur="fsbm_a_b", **kw):
    base = {"raw_pub_id": f"{chercheur}::{titre[:8]}", "chercheur_id": chercheur, "titre": titre, "auteurs": ["X"],
            "annee": 2021, "citations": 3, "abstract": None, "abstract_status": "not_found"}
    return {**base, **kw}


def test_clean_publication_record_keeps_raw_abstract_and_statuses():
    rec = clean_publication_record(_raw("A title", abstract="<b>Abstract</b> " + "Deep Learning " * 10,
                                        abstract_status="found", abstract_source="crossref", citations=-1,
                                        scholar_url="notaurl"))
    assert rec["abstract"].startswith("<b>Abstract</b>")          # brut conservé
    assert rec["abstract_clean"].startswith("deep learning")     # version NLP
    assert rec["citations"] is None and rec["scholar_url"] is None
    assert rec["embedding_source"] == "title_abstract"


def test_missing_abstract_stays_missing():
    rec = clean_publication_record(_raw("Only a title"))
    assert rec["abstract"] is None and rec["abstract_clean"] is None
    assert rec["abstract_status"] == "not_found" and rec["embedding_source"] == "title_only"


# ---------------------------------------------------------------- déduplication
def _clean(titre, chercheur="fsbm_a_b", **kw):
    return clean_publication_record(_raw(titre, chercheur, **kw))


def test_dedup_by_doi_and_shared_articles_keep_both_researchers():
    recs = [_clean("Machine learning for cancer", "fsbm_a_b", doi="10.1000/x1"),
            _clean("Machine Learning for Cancer (preprint)", "fsbm_c_d", doi="https://doi.org/10.1000/X1", annee=2022)]
    articles, links = deduplicate_publications(recs)
    assert len(articles) == 1
    assert {(l["chercheur_id"], l["article_id"]) for l in links} == {("fsbm_a_b", "art_00001"), ("fsbm_c_d", "art_00001")}
    assert articles[0]["chercheur_ids"] == ["fsbm_a_b", "fsbm_c_d"]


def test_dedup_by_normalized_title_and_year():
    recs = [_clean("Deep learning: a survey of methods", "fsbm_a_b"), _clean("DEEP LEARNING - A SURVEY OF METHODS!", "fsbm_c_d")]
    assert len(find_duplicate_groups(recs)) == 1


def test_dedup_does_not_merge_different_years_or_dois_or_short_titles():
    assert len(find_duplicate_groups([_clean("A study of solar cell efficiency", annee=2015),
                                      _clean("A study of solar cell efficiency", annee=2021)])) == 2
    assert len(find_duplicate_groups([_clean("Same title here for both", doi="10.1000/a"),
                                      _clean("Same title here for both", doi="10.1000/b")])) == 2
    assert len(find_duplicate_groups([_clean("Editorial"), _clean("Editorial", "fsbm_c_d")])) == 2


def test_fuzzy_dedup_is_prudent():
    close = [_clean("Machine learning algorithms for breast cancer prediction and diagnosis"),
             _clean("Machine learning algorithm for breast cancer prediction and diagnosis", "fsbm_c_d")]
    assert len(find_duplicate_groups(close, 0.93)) == 1
    different = [_clean("Machine learning algorithms for breast cancer prediction"),
                 _clean("Machine learning algorithms for lung cancer segmentation", "fsbm_c_d")]
    assert len(find_duplicate_groups(different, 0.93)) == 2


def test_fuzzy_dedup_never_merges_numbered_parts_or_volumes():
    parts = [_clean("Introduction to graph neural networks for molecules Part I"),
             _clean("Introduction to graph neural networks for molecules Part II", "fsbm_c_d")]
    assert len(find_duplicate_groups(parts, 0.90)) == 2
    numbered = [_clean("Deep learning for medical imaging segmentation 3", annee=2018),
                _clean("Deep learning for medical imaging segmentation 6", "fsbm_c_d", annee=2018)]
    assert len(find_duplicate_groups(numbered, 0.90)) == 2


def test_merge_keeps_max_citations_and_best_abstract():
    a = _clean("Neural networks for tumour detection", "fsbm_a_b", citations=5)
    b = _clean("Neural networks for tumour detection", "fsbm_c_d", citations=9, abstract="tumour detection " * 12,
               abstract_status="found", abstract_source="crossref", doi="10.1000/t")
    articles, _ = deduplicate_publications([a, b])
    assert articles[0]["citations"] == 9 and articles[0]["abstract_source"] == "crossref" and articles[0]["doi"] == "10.1000/t"


# ---------------------------------------------------------------- extraction du PDF (lignes de tableau)
HEADER = ["Etablissement", "Enseignant Chercheur", "Laboratoire", "Equipe", "Type Membre"]


def test_detect_header_mapping():
    assert detect_header_mapping(HEADER) == {0: "etablissement", 1: "chercheur_source_name", 2: "laboratoire",
                                             3: "equipe", 4: "type_membre"}
    assert detect_header_mapping(["FSBM", "DRISS.BOUGGAR", "Lab", "Eq", "Membre"]) == {}


def test_rows_to_records_and_dataframe():
    rows = [HEADER,
            ["FSBM", "DRISS.BOUGGAR", "Lab IA", "Eq 1", "Membre"],
            ["FSBM", "DRISS.BOUGGAR", "Lab Energie", None, "Membre associé"],   # même personne, 2e affectation
            ["FSBM", "DRISS.BOUGGAR", "Lab IA", "Eq 1", "Membre"],              # doublon exact
            ["ENCG", "SARA.ALAMI", "Lab X", "Eq 2", "Membre"],
            ["FSBM", None, "Lab IA", "Eq 1", "Membre"]]                         # ligne sans nom
    report = ExtractionReport(pdf="test.pdf")
    records, mapping = rows_to_records(rows, 1, {}, report)
    assert len(records) == 4 and mapping and report.skipped[0]["reason"] == "empty_name"
    df = build_members_dataframe(records)
    assert set(df["chercheur_id"]) == {"fsbm_driss_bouggar", "encg_sara_alami"}
    assert df["is_duplicate_row"].sum() == 1
    assert (df["chercheur_source_name"] == "DRISS.BOUGGAR").sum() == 3       # valeur originale conservée
    stats = summarize_members(df, "FSBM")
    assert stats["membres_fsbm_uniques"] == 1 and stats["personnes_uniques_total"] == 2
    assert stats["laboratoires_fsbm"] == 2 and stats["personnes_avec_plusieurs_affectations"] == 1
    (researcher,) = unique_researchers(df, "FSBM")
    assert researcher["laboratoire"] == "Lab IA | Lab Energie" and researcher["equipe"] == "Eq 1"


def test_dataset_can_be_restricted_to_researchers_with_a_scholar_id():
    from src.preprocessing.data_cleaner import clean_researchers

    records = [{"etablissement": "FSBM", "chercheur_source_name": n, "laboratoire": "L", "equipe": "E", "type_membre": "M"}
               for n in ("A.ONE", "B.TWO", "C.THREE")] + [
               {"etablissement": "ENCG", "chercheur_source_name": "D.FOUR", "laboratoire": "L", "equipe": "E", "type_membre": "M"}]
    members = build_members_dataframe(records)
    assert len(clean_researchers(members, [])) == 3                                   # tous les FSBM par défaut
    kept = clean_researchers(members, [], restrict_to={"fsbm_a_one", "fsbm_c_three", "encg_d_four"})
    assert kept["chercheur_id"].tolist() == ["fsbm_a_one", "fsbm_c_three"]           # ID connu ET établissement FSBM
    assert clean_researchers(members, [], restrict_to=set()).empty


def test_researchers_not_yet_collected_validate_cleanly_and_keep_their_manual_scholar_id():
    from src.preprocessing.data_cleaner import clean_researchers, validate_researchers

    members = build_members_dataframe([
        {"etablissement": "FSBM", "chercheur_source_name": "A.ONE", "laboratoire": "L", "equipe": "E", "type_membre": "M"},
        {"etablissement": "FSBM", "chercheur_source_name": "B.TWO", "laboratoire": "L", "equipe": "E", "type_membre": "M"}])
    df = clean_researchers(members, [], restrict_to={"fsbm_a_one", "fsbm_b_two"}, known_scholar_ids={"fsbm_a_one": "AbCdEf123456"})
    row = df.set_index("chercheur_id")
    assert row.loc["fsbm_a_one", "scholar_id"] == "AbCdEf123456"
    assert row.loc["fsbm_a_one", "scholar_url"] == "https://scholar.google.com/citations?user=AbCdEf123456"
    assert row.loc["fsbm_a_one", "scholar_profile_status"] == "pending"          # ID connu mais profil pas encore collecté
    assert validate_researchers(df) == []                                        # pas de NaN qui fait échouer Pydantic


def test_dataset_final_follows_the_professors_schema():
    import pandas as pd

    from src.preprocessing.data_cleaner import articles_to_dataframe, build_dataset_final, clean_researchers

    members = build_members_dataframe([{"etablissement": "FSBM", "chercheur_source_name": "A.ONE", "laboratoire": "Lab",
                                        "equipe": "Eq", "type_membre": "M"}])
    scholars = [{"chercheur_id": "fsbm_a_one", "scholar_id": "AbCdEf123456", "scholar_profile_status": "matched",
                 "profile": {"affiliation": "Faculty of science Ben M'Sik", "interests": ["NLP"],
                             "metrics": {"citations": 10, "h_index": 2, "i10_index": 1}, "since_year": 2021}}]
    researchers = clean_researchers(members, scholars)
    art = clean_publication_record({"raw_pub_id": "fsbm_a_one::p1", "chercheur_id": "fsbm_a_one", "titre": "A title",
                                    "auteurs": ["X"], "annee": 2021, "date_publication_raw": "2021/1/1", "citations": 5,
                                    "abstract": "deep learning " * 10, "abstract_status": "found"})
    articles, links = deduplicate_publications([art])
    dataset = build_dataset_final(researchers, articles_to_dataframe(articles), pd.DataFrame(links))
    (entry,) = dataset
    assert entry["chercheur_id"] == "AbCdEf123456"                 # « chercheur_id » = Scholar ID unique (sujet)
    assert entry["chercheur_id_interne"] == "fsbm_a_one" and entry["scholar_id"] == "AbCdEf123456"
    assert entry["metriques"]["citations_totales"] == 10 and entry["metriques"]["h_index"] == 2
    (article,) = entry["articles"]
    assert {"article_id", "titre", "auteurs", "date_publication", "journal", "citations", "abstract", "abstract_clean",
            "embedding_zembed1"} <= set(article)
    assert article["date_publication"] == "2021-01-01" and article["embedding_zembed1"] is None   # rempli par le script 04


def test_header_not_repeated_on_next_page_reuses_mapping():
    report = ExtractionReport(pdf="t")
    _, mapping = rows_to_records([HEADER, ["FSBM", "A.B", "L", "E", "M"]], 1, {}, report)
    records, _ = rows_to_records([["FSBM", "C.D", "L2", "E2", "M"]], 2, mapping, report)
    assert records[0]["laboratoire"] == "L2" and records[0]["source_page"] == 2


def test_forward_fill_merged_cells():
    records = [{"etablissement": "FSBM", "chercheur_source_name": "A.B", "laboratoire": "Lab 1", "equipe": None, "type_membre": None},
               {"etablissement": None, "chercheur_source_name": "C.D", "laboratoire": None, "equipe": None, "type_membre": None}]
    df = build_members_dataframe(records, forward_fill=["etablissement", "laboratoire"])
    assert df["etablissement"].tolist() == ["FSBM", "FSBM"] and df["laboratoire"].tolist() == ["Lab 1", "Lab 1"]
