"""Tests : import de Scholar ID saisis à la main (validation du format, rapprochement prudent, écriture du CSV)."""
import pytest

from src.data.scholar_ids import match_entries, read_overrides, validate_scholar_id, write_overrides

ROSTER = [{"chercheur_id": "fsbm_elhabib_benlahmar", "nom_complet": "Elhabib Benlahmar", "etablissement": "FSBM"},
          {"chercheur_id": "fsbm_nawal_sael", "nom_complet": "Nawal Sael", "etablissement": "FSBM"},
          {"chercheur_id": "fsbm_said_ouaskit", "nom_complet": "Said Ouaskit", "etablissement": "FSBM"},
          {"chercheur_id": "fsbm_aziza_elbakali", "nom_complet": "Aziza Elbakali", "etablissement": "FSBM"},
          {"chercheur_id": "fsbm_mohamed_bour", "nom_complet": "Mohamed Bour", "etablissement": "FSBM"},
          {"chercheur_id": "fsbm_mohamed_bouri", "nom_complet": "Mohamed Bouri", "etablissement": "FSBM"},
          {"chercheur_id": "flshbm_youssef_sefri", "nom_complet": "Youssef Sefri", "etablissement": "FLSHBM"}]


@pytest.mark.parametrize("value,ok", [("upOdTrEAAAAJ", True), ("-_RLt1UAAAAJ", True), ("YMFR4oAAAAJ", False),
                                      ("upOdTrEAAAAJX", False), ("up dTrEAAAAJ", False), ("", False)])
def test_scholar_id_format(value, ok):
    assert (validate_scholar_id(value) is None) is ok


def entry(name, sid):
    return {"nom_complet": name, "chercheur_id": sid}


def by_name(results):
    return {r.input_name: r for r in results}


def test_match_entries_statuses():
    res = by_name(match_entries([
        entry("Habib BEN LAHMAR", "upOdTrEAAAAJ"),          # nom collé dans le PDF
        entry("NAWAL SAEL", "hfAJHyYAAAAJ"),                 # casse / ordre
        entry("Said Ouaskit", "YMFR4oAAAAJ"),                # ID de 11 caractères
        entry("Aziza Elbakali Kassimi", "ADagKpAAAAAJ"),     # nom d'épouse en plus
        entry("Bouchaib Bounabat", "ErDjoI4AAAAJ"),          # absent du PDF
        entry("Sefri Youssef", "aHie7pgAAAAJ"),              # autre établissement
        entry("Mohamed Bou", "aaaaaaaaaaaa"),                # ambigu entre Bour et Bouri, ou non trouvé
    ], ROSTER))
    assert res["Habib BEN LAHMAR"].status == "accepted" and res["Habib BEN LAHMAR"].chercheur_id == "fsbm_elhabib_benlahmar"
    assert res["Habib BEN LAHMAR"].to_verify                                    # score < 0,95 : à relire
    assert res["NAWAL SAEL"].status == "accepted" and not res["NAWAL SAEL"].to_verify
    assert res["Said Ouaskit"].status == "invalid_id" and "11 caractères" in res["Said Ouaskit"].note
    assert res["Aziza Elbakali Kassimi"].status == "accepted" and res["Aziza Elbakali Kassimi"].to_verify
    assert res["Bouchaib Bounabat"].status == "not_in_roster" and res["Bouchaib Bounabat"].chercheur_id is None
    assert res["Sefri Youssef"].status == "other_institution"
    assert res["Mohamed Bou"].status != "accepted"                              # jamais de choix arbitraire


def test_same_person_repeated_with_same_id_is_fine_but_conflicting_ids_are_rejected():
    res = match_entries([entry("Nawal Sael", "hfAJHyYAAAAJ"), entry("NAWAL SAEL", "hfAJHyYAAAAJ"),
                         entry("Nawal Sael", "ZZZZZZZZZZZZ")], ROSTER)
    assert [r.status for r in res] == ["accepted", "accepted", "conflict"]


def test_write_overrides_preserves_existing_ids_and_only_adds_new_ones(tmp_path):
    records = [{"chercheur_id": "fsbm_nawal_sael", "nom_complet": "Nawal Sael", "laboratoire": "Labo A", "equipe": None},
               {"chercheur_id": "fsbm_said_ouaskit", "nom_complet": "Said Ouaskit", "laboratoire": None, "equipe": "Eq"}]
    path = tmp_path / "overrides.csv"
    assert write_overrides(records, path) == 0
    assert write_overrides(records, path, {"fsbm_nawal_sael": "hfAJHyYAAAAJ"}) == 1
    assert write_overrides(records, path) == 1                                  # relancer sans nouvel ID ne perd rien
    assert read_overrides(path) == {"fsbm_nawal_sael": "hfAJHyYAAAAJ", "fsbm_said_ouaskit": ""}
