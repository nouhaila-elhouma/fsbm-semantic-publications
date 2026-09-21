"""Tests : lecture hors ligne de pages de profil Scholar enregistrées à la main (aucun accès réseau)."""
from test_scholar import PROFILE_HTML

from src.data.scholar_saved_pages import extract_scholar_id, inspect_saved_page, read_html, state_from_saved_page

ROBOT_PAGE = ("<html><head><style>" + "a{color:red}" * 8000 + "</style></head><body>Google Scholar Loading... The system can't perform "
              "the operation now. Please show you're not a robot</body></html>")


def test_scholar_id_is_read_from_publication_links():
    assert extract_scholar_id(PROFILE_HTML) == "AAA111"
    assert extract_scholar_id("<a href='/citations?user=ZZZZZZZZZZZZ&hl=en'>x</a>") == "ZZZZZZZZZZZZ"
    assert extract_scholar_id("<html>rien</html>") is None


def test_saved_pages_are_diagnosed_without_any_network_access():
    ok = inspect_saved_page(PROFILE_HTML)
    assert ok["status"] == "ok" and ok["scholar_id"] == "AAA111" and ok["profile"]["metrics"]["h_index"] == 25
    robot = inspect_saved_page(ROBOT_PAGE)
    assert robot["status"] == "robot_check" and "réenregistrez" in robot["detail"]
    assert inspect_saved_page("<html><body>un autre site</body></html>")["status"] == "not_a_profile"


def test_state_from_saved_page_is_labelled_and_capped():
    info = inspect_saved_page(PROFILE_HTML)
    base = {"chercheur_id": "fsbm_h_b", "nom_complet": "Habib B", "scholar_profile_status": "error", "publications": []}
    state = state_from_saved_page(base, "AAA111", info["profile"], max_publications=1, source_file="Habib - Google Scholar.html")
    assert state["scholar_profile_status"] == "matched" and state["match_method"] == "manual_saved_html"
    assert state["data_source"] == "google_scholar" and state["collection_status"] == "complete"
    assert state["acquisition"]["method"] == "saved_html" and state["acquisition"]["file"].endswith(".html")
    assert len(state["publications"]) == 1                                  # plafond respecté
    pub = state["publications"][0]
    assert pub["titre"].startswith("Machine learning algorithms") and pub["citations"] == 507
    assert pub["detail_status"] == "skipped" and pub["chercheur_id"] == "fsbm_h_b" and pub["scholar_id"] == "AAA111"
    assert state["profile"]["metrics"]["citations"] == 2819


def test_read_html_tolerates_bad_encoding(tmp_path):
    f = tmp_path / "p.html"
    f.write_bytes("<html>é ".encode("utf-8") + b"\xff\xfe</html>")
    assert "é" in read_html(f)                                              # pas d'exception sur un octet invalide
