"""Tests : parsing Scholar, matching prudent de profils, robustesse du collecteur (blocage, reprise, erreurs).

Les fixtures HTML reproduisent la structure des pages Scholar ; la page de détail reprend les métadonnées de
l'exemple fourni dans le sujet (publication « Machine learning algorithms for breast cancer… »).
Aucun test ne contacte Google Scholar.
"""
from __future__ import annotations

import copy

import pytest

from src.data.robots import RobotsDisallowedError, RobotsRules, SearchNotAllowedError
from src.utils.config import load_settings
from src.data.scholar_matching import decide_match, name_match_score, score_candidate
from src.data.scholar_parsing import (ScholarParseError, detect_block, parse_profile, parse_publication_detail,
                                      parse_search_results)
from src.data.scholar_scraper import (HttpScholarBackend, ScholarBackend, ScholarCollector, make_backend,
                                      merge_publication_detail, new_publication_record)
from src.utils.io import read_json
from src.utils.retry import BlockedError, TransientError

SEARCH_HTML = """
<div class="gsc_1usr"><h3 class="gs_ai_name"><a href="/citations?hl=en&user=AAA111">Habib BEN LAHMAR</a></h3>
 <div class="gs_ai_aff">Professor in computer science at HASSAN 2 University, Casablanca Faculty of science Ben M'Sik</div>
 <div class="gs_ai_eml">Verified email at univh2c.ma</div><div class="gs_ai_cby">Cited by 2819</div>
 <div class="gs_ai_int"><a class="gs_ai_one_int">NLP</a><a class="gs_ai_one_int">deep learning</a></div></div>
<div class="gsc_1usr"><h3 class="gs_ai_name"><a href="/citations?user=BBB222">Habib Benlahmar</a></h3>
 <div class="gs_ai_aff">University of Elsewhere</div></div>
"""

PROFILE_HTML = """
<div id="gsc_prf_i"><div id="gsc_prf_in">Habib BEN LAHMAR</div>
 <div class="gsc_prf_il">Professor in computer science at HASSAN 2 University, Casablanca Faculty of science Ben M'Sik</div>
 <div class="gsc_prf_il" id="gsc_prf_ivh">Verified email at univh2c.ma</div>
 <div id="gsc_prf_int"><a>Moteurs de recherche</a><a>Web sémantique</a><a>NLP</a></div></div>
<table id="gsc_rsb_st"><thead><tr><th></th><th>All</th><th>Since 2021</th></tr></thead><tbody>
 <tr><td class="gsc_rsb_sc1">Citations</td><td class="gsc_rsb_std">2819</td><td class="gsc_rsb_std">2329</td></tr>
 <tr><td class="gsc_rsb_sc1">h-index</td><td class="gsc_rsb_std">25</td><td class="gsc_rsb_std">21</td></tr>
 <tr><td class="gsc_rsb_sc1">i10-index</td><td class="gsc_rsb_std">63</td><td class="gsc_rsb_std">51</td></tr></tbody></table>
<div id="gsc_rsb_co"><a href="/citations?user=CO1">Sanaa El Filali</a><a href="/citations?user=CO2">Someone Else</a></div>
<table><tbody>
 <tr class="gsc_a_tr"><td class="gsc_a_t"><a class="gsc_a_at" href="/citations?view_op=view_citation&user=AAA111&citation_for_view=AAA111:p1">Machine learning algorithms for breast cancer prediction and diagnosis</a>
   <div class="gs_gray">MA Naji, S El Filali, K Aarika, ELH Benlahmar, R Ait Abdelouahid, ...</div><div class="gs_gray">Procedia computer science 191, 487-492, 2021</div></td>
   <td class="gsc_a_c"><a class="gsc_a_ac">507</a></td><td class="gsc_a_y"><span class="gsc_a_h">2021</span></td></tr>
 <tr class="gsc_a_tr"><td class="gsc_a_t"><a class="gsc_a_at" href="/citations?view_op=view_citation&user=AAA111&citation_for_view=AAA111:p2">An uncited paper</a>
   <div class="gs_gray">H Benlahmar</div><div class="gs_gray"></div></td>
   <td class="gsc_a_c"><a class="gsc_a_ac"> </a></td><td class="gsc_a_y"><span class="gsc_a_h"></span></td></tr>
</tbody></table><button id="gsc_bpf_more" disabled>Show more</button>
"""

DETAIL_HTML = """
<div id="gsc_oci_title_wrapper"><div id="gsc_oci_title"><a class="gsc_oci_title_link" href="https://www.sciencedirect.com/science/article/pii/S1877050921011455">Machine learning algorithms for breast cancer prediction and diagnosis</a></div>
 <div id="gsc_oci_title_gg"><div class="gsc_oci_title_ggi"><a href="https://www.sciencedirect.com/science/article/pii/S1877050921011455/pdf">[PDF] sciencedirect.com</a></div></div></div>
<div id="gsc_oci_table">
 <div class="gs_scl"><div class="gsc_oci_field">Authors</div><div class="gsc_oci_value">Mohammed Amine Naji, Sanaa El Filali, Kawtar Aarika, EL Habib Benlahmar, Rachida Ait Abdelouahid, Olivier Debauche</div></div>
 <div class="gs_scl"><div class="gsc_oci_field">Publication date</div><div class="gsc_oci_value">2021/1/1</div></div>
 <div class="gs_scl"><div class="gsc_oci_field">Journal</div><div class="gsc_oci_value">Procedia computer science</div></div>
 <div class="gs_scl"><div class="gsc_oci_field">Volume</div><div class="gsc_oci_value">191</div></div>
 <div class="gs_scl"><div class="gsc_oci_field">Pages</div><div class="gsc_oci_value">487-492</div></div>
 <div class="gs_scl"><div class="gsc_oci_field">Publisher</div><div class="gsc_oci_value">Elsevier</div></div>
 <div class="gs_scl"><div class="gsc_oci_field">Description</div><div class="gsc_oci_value"><div class="gsh_csp">Each year number of deaths is increasing extremely because of breast cancer. A performance evaluation and comparison is carried out …</div></div></div>
</div>
"""


# ---------------------------------------------------------------- parsing
def test_parse_search_results():
    cands = parse_search_results(SEARCH_HTML)
    assert [c["scholar_id"] for c in cands] == ["AAA111", "BBB222"]
    assert cands[0]["email_domain"] == "univh2c.ma" and cands[0]["cited_by"] == 2819
    assert cands[0]["interests"] == ["NLP", "deep learning"] and cands[1]["email_domain"] is None


def test_parse_profile_metrics_and_publications():
    p = parse_profile(PROFILE_HTML)
    assert p["name"] == "Habib BEN LAHMAR" and p["email_domain"] == "univh2c.ma"
    assert p["interests"] == ["Moteurs de recherche", "Web sémantique", "NLP"]
    assert p["metrics"] == {"citations": 2819, "citations_since": 2329, "h_index": 25, "h_index_since": 21,
                            "i10_index": 63, "i10_index_since": 51}
    assert p["since_year"] == 2021 and p["has_more"] is False
    assert p["coauthors"] == ["Sanaa El Filali", "Someone Else"]
    first, second = p["publications"]
    assert first["scholar_pub_id"] == "AAA111:p1" and first["citations"] == 507 and first["annee"] == 2021
    assert second["citations"] == 0 and second["annee"] is None       # cellule vide Scholar = 0 citation ; année inconnue = None


def test_parse_profile_raises_on_unexpected_html():
    with pytest.raises(ScholarParseError):
        parse_profile("<html><body>rien</body></html>")


def test_parse_publication_detail_matches_documented_example():
    d = parse_publication_detail(DETAIL_HTML)
    assert d["titre"].startswith("Machine learning algorithms for breast cancer")
    assert d["auteurs"][0] == "Mohammed Amine Naji" and d["auteurs"][-1] == "Olivier Debauche" and len(d["auteurs"]) == 6
    assert (d["date_publication_raw"], d["journal"], d["volume"], d["pages"], d["publisher"]) == \
           ("2021/1/1", "Procedia computer science", "191", "487-492", "Elsevier")
    assert d["pdf_url"].endswith("/pdf") and d["abstract"].endswith("carried out …")


def test_truncated_scholar_abstract_is_flagged_not_hidden():
    state = {"chercheur_id": "fsbm_h_b", "scholar_id": "AAA111"}
    pub = new_publication_record(state, parse_profile(PROFILE_HTML)["publications"][0], True)
    merge_publication_detail(pub, parse_publication_detail(DETAIL_HTML))
    assert pub["abstract_status"] == "truncated" and pub["abstract_source"] == "google_scholar"
    assert pub["auteurs_truncated"] is False and pub["detail_status"] == "done"


@pytest.mark.parametrize("status,url,text", [(429, "https://scholar.google.com/citations", ""),
                                             (200, "https://www.google.com/sorry/index?continue=x", ""),
                                             (200, "https://scholar.google.com/citations", "Our systems have detected unusual traffic"),
                                             (200, "https://scholar.google.com/citations", '<form id="gs_captcha_f">')])
def test_block_detection(status, url, text):
    with pytest.raises(BlockedError):
        detect_block(status, url, text)


def test_robot_check_served_with_http_200_after_a_long_page_head_is_detected():
    """Cas réel observé : « Please show you're not a robot » (HTTP 200) noyé après ~70 Ko de CSS/JS."""
    page = "<html><head><style>" + "a{color:red}" * 8000 + "</style></head><body><div>The system can't perform the operation now. "            "Try again later.</div><div>Please show you're not a robot</div></body></html>"
    assert len(page) > 60000
    with pytest.raises(BlockedError):
        detect_block(200, "https://scholar.google.com/citations?user=X", page)


def test_normal_page_is_not_a_block():
    detect_block(200, "https://scholar.google.com/citations", PROFILE_HTML)


# ---------------------------------------------------------------- matching
def test_name_match_score_variants():
    assert name_match_score("Habib Ben Lahmar", "HABIB BEN LAHMAR") == 1.0
    assert name_match_score("Ben Lahmar Habib", "Habib Ben Lahmar") == 1.0                  # ordre
    assert name_match_score("Habib Ben Lahmar", "EL Habib Benlahmar") >= 0.9                # espacement + particule
    assert name_match_score("Sara Alami", "S Alami") >= 0.66                                # initiale
    # noms collés dans le PDF FSBM vs écriture Scholar
    assert name_match_score("Elhabib Benlahmar", "El Habib Ben Lahmar") >= 0.95
    assert name_match_score("Mohammed Aitdaoud", "Mohammed Ait Daoud") >= 0.95
    assert name_match_score("Hassaniahmed Adlouni", "Hassani Ahmed Adlouni") >= 0.95
    assert name_match_score("Rachida Aitabdelouahid", "Rachida Ait Abdelouahid") >= 0.95
    assert name_match_score("Mohamed Bour", "Mohamed Talea") < 0.6
    # particule « El » collée au prénom, et nom d'épouse ajouté : acceptés mais < 0,95 (donc « à vérifier »)
    assert 0.9 <= name_match_score("Habib BEN LAHMAR", "Elhabib Benlahmar") < 0.95
    assert 0.8 <= name_match_score("Aziza Elbakali Kassimi", "Aziza Elbakali") < 0.95
    assert name_match_score("Sara Alami", "Sara") < 0.8                                      # un seul mot commun : jamais accepté
    assert name_match_score("Sara Alami", "Karim Tazi") < 0.4


RESEARCHER = {"chercheur_id": "fsbm_habib_ben_lahmar", "nom_complet": "Habib Ben Lahmar", "laboratoire": "Laboratoire Informatique"}


def test_name_alone_is_never_enough():
    only_name = {"scholar_id": "X", "name": "Habib Ben Lahmar", "affiliation": "Some Other University", "email_domain": None, "interests": []}
    assert score_candidate(RESEARCHER, only_name).score <= 0.40 + 1e-9
    decision = decide_match(RESEARCHER, [only_name])
    assert decision.status == "not_found" and decision.scholar_id is None


def test_matched_with_affiliation_evidence():
    cands = parse_search_results(SEARCH_HTML)
    decision = decide_match(RESEARCHER, cands)
    assert decision.status == "matched" and decision.scholar_id == "AAA111" and decision.confidence >= 0.7
    assert len(decision.candidates) == 2 and decision.candidates[0]["scholar_id"] == "AAA111"


def test_two_equally_plausible_profiles_are_ambiguous_and_kept_for_review():
    twin = {"scholar_id": "AAA111", "name": "Habib Ben Lahmar", "affiliation": "Faculty of science Ben M'Sik, Hassan II University, Casablanca",
            "email_domain": "univh2c.ma", "interests": []}
    decision = decide_match(RESEARCHER, [twin, {**twin, "scholar_id": "ZZZ999"}])
    assert decision.status == "ambiguous" and decision.scholar_id is None and len(decision.candidates) == 2


def test_no_results_is_not_found():
    d = decide_match(RESEARCHER, [])
    assert d.status == "not_found" and d.candidates == []


# ---------------------------------------------------------------- backend HTTP (session simulée)
class FakeResponse:
    def __init__(self, status, text="", url="https://scholar.google.com/citations"):
        self.status_code, self.text, self.url = status, text, url

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)


class FakeSession:
    def __init__(self, responses):
        self.responses, self.calls, self.headers, self.requests = list(responses), 0, {}, []

    def get(self, url, params=None, **k):
        self.calls += 1
        self.requests.append((url, params))
        return self.responses.pop(0)


def test_http_backend_stops_immediately_on_429_without_retry():
    session = FakeSession([FakeResponse(429), FakeResponse(200, SEARCH_HTML)])
    backend = HttpScholarBackend("ua", retries=3, backoff_base=0, delay_fn=lambda: None, session=session, respect_robots=False)
    with pytest.raises(BlockedError):
        backend.search_authors("x")
    assert session.calls == 1                                   # jamais de nouvelle tentative après un blocage


def test_http_backend_retries_transient_errors_with_backoff():
    session = FakeSession([FakeResponse(503), FakeResponse(503), FakeResponse(200, SEARCH_HTML)])
    backend = HttpScholarBackend("ua", retries=3, backoff_base=0, delay_fn=lambda: None, session=session, respect_robots=False)
    assert len(backend.search_authors("x")) == 2 and session.calls == 3


def test_http_backend_gives_up_after_limited_retries():
    session = FakeSession([FakeResponse(500)] * 10)
    backend = HttpScholarBackend("ua", retries=2, backoff_base=0, delay_fn=lambda: None, session=session, respect_robots=False)
    with pytest.raises(TransientError):
        backend.search_authors("x")
    assert session.calls == 3                                   # 1 essai + 2 retries, pas plus


def test_http_backend_paginates_until_max_publications():
    page = PROFILE_HTML.replace('<button id="gsc_bpf_more" disabled>', '<button id="gsc_bpf_more">')
    session = FakeSession([FakeResponse(200, page), FakeResponse(200, page)])
    backend = HttpScholarBackend("ua", delay_fn=lambda: None, session=session, respect_robots=False)
    author = backend.fetch_author("AAA111", max_publications=3)
    assert len(author["publications"]) == 3 and session.calls == 2


# ---------------------------------------------------------------- conformité robots.txt (règles réelles de Google Scholar)
SCHOLAR_ROBOTS = """
User-agent: *
Disallow: /citations?
Allow: /citations?user=
Disallow: /citations?*cstart=
Disallow: /citations?user=*%40
Disallow: /citations?user=*@
Allow: /citations?view_op=new_profile
Allow: /citations?view_op=top_venues
"""


def test_robots_rules_match_google_semantics():
    rules = RobotsRules.from_text(SCHOLAR_ROBOTS)
    assert rules.is_allowed("/citations?user=AAA111&hl=en&pagesize=50")                       # profil : autorisé
    assert rules.is_allowed("/citations?user=AAA111&view_op=view_citation&citation_for_view=AAA111%3Ap1&hl=en")
    assert not rules.is_allowed("/citations?user=AAA111&hl=en&cstart=100&pagesize=100")       # pagination : interdite
    assert not rules.is_allowed("/citations?view_op=search_authors&mauthors=habib&hl=en")     # recherche : interdite
    assert not rules.is_allowed("/citations?view_op=view_citation&user=AAA111")               # « view_op » en tête : interdit
    assert not rules.is_allowed("/citations?user=AAA111%40univh2c.ma")                        # adresse e-mail : interdit
    assert rules.is_allowed("/citations?view_op=top_venues")
    assert rules.is_allowed("/about")                                                        # hors /citations : rien d'interdit ici


def test_robots_longest_rule_wins_and_allow_wins_ties():
    rules = RobotsRules.from_text("User-agent: *\nDisallow: /a\nAllow: /a/b\nDisallow: /c$\nAllow: /c$")
    assert rules.is_allowed("/a/b/x") and not rules.is_allowed("/a/z")
    assert rules.is_allowed("/c")                                                            # égalité : Allow gagne


def test_backend_refuses_author_search_by_default_without_any_request():
    session = FakeSession([FakeResponse(200, SEARCH_HTML)])
    backend = HttpScholarBackend("ua", delay_fn=lambda: None, session=session, robots_text=SCHOLAR_ROBOTS)
    assert backend.search_allowed is False
    with pytest.raises(SearchNotAllowedError):
        backend.search_authors("Habib Ben Lahmar")
    assert session.calls == 0                                                                # rien n'est envoyé


def test_author_search_requires_explicit_opt_in():
    session = FakeSession([FakeResponse(200, SEARCH_HTML)])
    backend = HttpScholarBackend("ua", delay_fn=lambda: None, session=session, robots_text=SCHOLAR_ROBOTS, allow_search=True)
    assert backend.search_allowed is True and len(backend.search_authors("x")) == 2 and session.calls == 1


def test_profile_and_detail_urls_are_built_in_the_allowed_form_without_cstart():
    page = PROFILE_HTML.replace('<button id="gsc_bpf_more" disabled>', '<button id="gsc_bpf_more">')
    session = FakeSession([FakeResponse(200, page), FakeResponse(200, DETAIL_HTML)])
    backend = HttpScholarBackend("ua", delay_fn=lambda: None, session=session, robots_text=SCHOLAR_ROBOTS)
    author = backend.fetch_author("AAA111", max_publications=50)
    backend.fetch_publication_detail("AAA111", "AAA111:p1")
    assert session.calls == 2                                                                # pas de 2e page malgré « Show more »
    (_, p1), (_, p2) = session.requests
    assert list(p1)[0] == "user" and "cstart" not in p1
    assert list(p2)[0] == "user" and p2["view_op"] == "view_citation"
    assert len(author["publications"]) == 2


def test_robots_violation_is_never_sent():
    session = FakeSession([FakeResponse(200, "")])
    backend = HttpScholarBackend("ua", delay_fn=lambda: None, session=session, robots_text=SCHOLAR_ROBOTS)
    with pytest.raises(RobotsDisallowedError):
        backend._get({"view_op": "view_citation", "user": "AAA111"})
    assert session.calls == 0


def test_robots_txt_is_fetched_once_and_unreadable_robots_fails_closed():
    session = FakeSession([FakeResponse(200, SCHOLAR_ROBOTS), FakeResponse(200, PROFILE_HTML)])
    backend = HttpScholarBackend("ua", delay_fn=lambda: None, session=session)
    backend.fetch_author("AAA111", 5)
    assert session.requests[0][0].endswith("/robots.txt") and session.calls == 2
    closed = HttpScholarBackend("ua", delay_fn=lambda: None, session=FakeSession([FakeResponse(500, "")]))
    with pytest.raises(RobotsDisallowedError):
        closed.fetch_author("AAA111", 5)


def test_scholarly_backend_is_refused_while_robots_are_respected():
    cfg = load_settings()
    cfg["scraping"]["backend"] = "scholarly"
    with pytest.raises(ValueError, match="robots"):
        make_backend(cfg)


# ---------------------------------------------------------------- collecteur : cache, checkpoints, reprise
@pytest.fixture
def settings():
    cfg = load_settings()
    cfg["scraping"]["checkpoint_every"] = 1
    cfg["scraping"]["fetch_publication_details"] = True        # les tests couvrent le mode complet ; le défaut réel est « profil seul »
    return cfg


class FakeBackend(ScholarBackend):
    """Backend en mémoire : 3 chercheurs, comportements programmables (blocage, erreur, inconnu)."""

    name = "fake"

    def __init__(self, block_on_detail_after=None, fail_names=()):
        self.calls = {"search": 0, "author": 0, "detail": 0}
        self.block_on_detail_after, self.fail_names = block_on_detail_after, set(fail_names)

    def search_authors(self, query):
        self.calls["search"] += 1
        name = query.replace(" Hassan II University", "")
        if name in self.fail_names:
            raise RuntimeError("HTML inattendu")
        if name == "Inconnu Total":
            return []
        sid = "ID_" + name.replace(" ", "_")
        return [{"scholar_id": sid, "name": name, "affiliation": "Faculty of science Ben M'Sik, Hassan II University, Casablanca",
                 "email_domain": "univh2c.ma", "interests": [], "cited_by": 1, "scholar_url": f"https://scholar.google.com/citations?user={sid}"}]

    def fetch_author(self, scholar_id, max_publications):
        self.calls["author"] += 1
        pubs = [{"scholar_pub_id": f"{scholar_id}:p{i}", "titre": f"Paper {i} of {scholar_id}", "auteurs_listing": "A B, C D, ...",
                 "venue_raw": "J 1, 2020", "annee": 2020, "citations": i, "scholar_url": "https://scholar.google.com/x"} for i in range(3)]
        return {"name": scholar_id, "affiliation": "FSBM", "email_domain": "univh2c.ma", "interests": ["NLP"], "since_year": 2021,
                "metrics": {"citations": 10, "citations_since": 5, "h_index": 2, "h_index_since": 1, "i10_index": 0, "i10_index_since": 0},
                "coauthors": [], "publications": pubs[:max_publications], "has_more": False}

    def fetch_publication_detail(self, scholar_id, scholar_pub_id):
        self.calls["detail"] += 1
        if self.block_on_detail_after is not None and self.calls["detail"] > self.block_on_detail_after:
            raise BlockedError("CAPTCHA")
        return {"titre": "t", "auteurs": ["A B", "C D"], "abstract": "A complete abstract about deep learning. " * 3, "journal": "J"}


ROSTER = [{"chercheur_id": "fsbm_alice_martin", "nom_complet": "Alice Martin", "laboratoire": "L1", "equipe": None},
          {"chercheur_id": "fsbm_inconnu_total", "nom_complet": "Inconnu Total", "laboratoire": "L1", "equipe": None},
          {"chercheur_id": "fsbm_bob_durand", "nom_complet": "Bob Durand", "laboratoire": "L2", "equipe": None}]


def make_collector(backend, settings, tmp_path, **kw):
    return ScholarCollector(backend, settings, tmp_path / "raw", tmp_path / "cache", **kw)


def test_full_run_writes_cache_and_consolidated_files(settings, tmp_path):
    collector = make_collector(FakeBackend(), settings, tmp_path)
    summary = collector.run(ROSTER, max_publications=2)
    assert summary["blocked"] is False and summary["profile_status_counts"] == {"matched": 2, "not_found": 1}
    assert len(list((tmp_path / "cache" / "scholar").glob("*.json"))) == 3         # 1 fichier de cache par chercheur
    scholars, pubs = read_json(tmp_path / "raw" / "scholars_raw.json"), read_json(tmp_path / "raw" / "publications_raw.json")
    assert len(scholars) == 3 and len(pubs) == 4                                   # 2 chercheurs × 2 publications
    assert all(p["abstract_status"] == "found" and p["detail_status"] == "done" for p in pubs)
    missing = next(s for s in scholars if s["chercheur_id"] == "fsbm_inconnu_total")
    assert missing["scholar_profile_status"] == "not_found" and missing["profile"] is None and missing["scholar_id"] is None


def test_block_stops_cleanly_keeps_data_and_resume_finishes_the_job(settings, tmp_path):
    blocking = FakeBackend(block_on_detail_after=4)                       # 3 détails OK (1er chercheur) + 1 puis blocage
    first = make_collector(blocking, settings, tmp_path).run(ROSTER, 3)
    assert first["blocked"] is True and first["blocked_at"] == "fsbm_bob_durand"
    saved = read_json(tmp_path / "raw" / "scholars_raw.json")
    by_id = {s["chercheur_id"]: s for s in saved}
    assert by_id["fsbm_alice_martin"]["collection_status"] == "complete"      # rien n'est perdu
    assert by_id["fsbm_bob_durand"]["collection_status"] == "blocked" and by_id["fsbm_bob_durand"]["n_publications_collected"] == 3

    healthy = FakeBackend()
    second = make_collector(healthy, settings, tmp_path).run(ROSTER, 3, resume=True)
    assert second["blocked"] is False and second["skipped_done"] == 2         # Alice + inconnu déjà faits
    assert healthy.calls["search"] == 0 and healthy.calls["author"] == 0      # profil déjà connu : aucune requête refaite
    assert healthy.calls["detail"] == 2                                       # seuls les 2 détails manquants de Bob
    final = read_json(tmp_path / "raw" / "publications_raw.json")
    assert len(final) == 6 and all(p["detail_status"] == "done" for p in final)


def test_error_on_one_researcher_does_not_stop_the_others(settings, tmp_path):
    collector = make_collector(FakeBackend(fail_names={"Alice Martin"}), settings, tmp_path)
    summary = collector.run(ROSTER, 2)
    assert summary["blocked"] is False
    assert summary["profile_status_counts"] == {"error": 1, "not_found": 1, "matched": 1}
    state = collector.states["fsbm_alice_martin"]
    assert state["scholar_profile_status"] == "error" and "HTML inattendu" in state["status_detail"]


def test_errors_are_retried_on_resume(settings, tmp_path):
    make_collector(FakeBackend(fail_names={"Alice Martin"}), settings, tmp_path).run(ROSTER, 2)
    healthy = FakeBackend()
    summary = make_collector(healthy, settings, tmp_path).run(ROSTER, 2, resume=True)
    assert summary["profile_status_counts"]["matched"] == 2
    assert summary["skipped_done"] == 2                                     # Bob et « Inconnu » déjà terminés
    assert healthy.calls["search"] == 1 and healthy.calls["author"] == 1   # seule Alice (en erreur) est retentée


def test_manual_override_replaces_ambiguous_status(settings, tmp_path):
    class Twin(FakeBackend):
        def search_authors(self, query):
            base = super().search_authors(query)
            return base + [{**base[0], "scholar_id": "OTHER"}] if base else base

    make_collector(Twin(), settings, tmp_path).run(ROSTER[:1], 2)
    first = read_json(tmp_path / "raw" / "scholars_raw.json")[0]
    assert first["scholar_profile_status"] == "ambiguous" and len(first["candidates"]) == 2
    assert (tmp_path / "raw" / "scholar_candidates_review.csv").read_text(encoding="utf-8").count("fsbm_alice_martin") == 2

    overrides = tmp_path / "overrides.csv"
    overrides.write_text("chercheur_id,scholar_id\nfsbm_alice_martin,ID_Alice_Martin\n", encoding="utf-8")
    second = make_collector(Twin(), settings, tmp_path, overrides_path=overrides).run(ROSTER[:1], 2, resume=True)
    state = read_json(tmp_path / "raw" / "scholars_raw.json")[0]
    assert second["skipped_done"] == 0 and state["scholar_profile_status"] == "matched"
    assert state["match_method"] == "manual_override" and state["profile_match_confidence"] == 1.0


def test_no_resume_rescrapes_but_resume_does_not(settings, tmp_path):
    make_collector(FakeBackend(), settings, tmp_path).run(ROSTER[:1], 2)
    again = FakeBackend()
    make_collector(again, settings, tmp_path).run(ROSTER[:1], 2, resume=True)
    assert again.calls["search"] == 0
    fresh = FakeBackend()
    make_collector(fresh, settings, tmp_path).run(ROSTER[:1], 2, resume=False)
    assert fresh.calls["search"] >= 1


def test_collector_only_processes_known_scholar_ids_when_search_is_disabled(settings, tmp_path):
    class NoSearch(FakeBackend):
        search_allowed = False

    overrides = tmp_path / "ids.csv"
    overrides.write_text("chercheur_id,scholar_id,nom_complet\nfsbm_alice_martin,ID_Alice_Martin,Alice Martin\n"
                         "fsbm_bob_durand,,Bob Durand\n", encoding="utf-8")                # ligne vide = ID non renseigné
    backend = NoSearch()
    summary = make_collector(backend, settings, tmp_path, overrides_path=overrides).run(ROSTER, 2)
    assert summary["processed"] == 1 and summary["needs_scholar_id"] == 2                     # Bob + « Inconnu Total »
    assert backend.calls["search"] == 0 and backend.calls["author"] == 1                      # aucune recherche, profil direct
    scholars = read_json(tmp_path / "raw" / "scholars_raw.json")
    assert [s["chercheur_id"] for s in scholars] == ["fsbm_alice_martin"]
    assert scholars[0]["match_method"] == "manual_override" and scholars[0]["collection_status"] == "complete"


def test_switching_to_profile_only_mode_after_a_block_finishes_the_researcher(settings, tmp_path):
    """Un chercheur bloqué en plein détail (pubs « pending ») est terminé proprement en mode « profil seul »."""
    blocking = FakeBackend(block_on_detail_after=1)
    first = make_collector(blocking, settings, tmp_path).run(ROSTER[:1], 3)
    assert first["blocked"] is True
    cfg = copy.deepcopy(settings)
    cfg["scraping"]["fetch_publication_details"] = False
    healthy = FakeBackend()
    second = make_collector(healthy, cfg, tmp_path).run(ROSTER[:1], 3, resume=True)
    state = read_json(tmp_path / "raw" / "scholars_raw.json")[0]
    assert second["blocked"] is False and healthy.calls["detail"] == 0 and state["collection_status"] == "complete"
    pubs = read_json(tmp_path / "raw" / "publications_raw.json")
    assert {p["detail_status"] for p in pubs} == {"done", "skipped"}


def test_no_details_mode_skips_publication_pages(settings, tmp_path):
    cfg = copy.deepcopy(settings)
    cfg["scraping"]["fetch_publication_details"] = False
    backend = FakeBackend()
    make_collector(backend, cfg, tmp_path).run(ROSTER[:1], 2)
    assert backend.calls["detail"] == 0
    pubs = read_json(tmp_path / "raw" / "publications_raw.json")
    assert all(p["detail_status"] == "skipped" and p["abstract"] is None and p["abstract_status"] == "not_found" for p in pubs)
