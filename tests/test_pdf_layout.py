"""Tests : extraction par coordonnées de mots (impression web « Membres Uh2c »).

La géométrie des mots reprend des mesures faites sur le vrai PDF (colonnes x0 ≈ 103 / 129 / 214 / 362 / 517 / 702 pt,
cellules de 2-3 lignes centrées verticalement sur le numéro de ligne, barre latérale à gauche, pied de page « Exporter »).
Un test d'intégration s'exécute sur le vrai PDF s'il est présent dans data/input/ (fichier non versionné).
"""
from pathlib import Path

import pytest

from src.data.extract_fsbm_members import (ExtractionReport, build_members_dataframe, check_row_sequence,
                                           join_wrapped_lines, run_extraction, summarize_members, words_to_records)

LINE_H = 10.9
X = {"#": 103.3, "etab": 129.3, "nom": 213.9, "lab": 362.4, "eq": 517.3, "type": 701.6}


def w(text, x0, top):
    return {"text": text, "x0": x0, "x1": x0 + 6 * len(text), "top": top, "bottom": top + LINE_H}


def header(top=140.6):
    return [w("#", X["#"], top), w("Etablissement", X["etab"], top), w("Enseignant", X["nom"], top),
            w("Chercheur", 280.0, top), w("Laboratoire", X["lab"], top), w("Equipe", X["eq"], top),
            w("Type", X["type"], top), w("Membre", 735.0, top)]


def row(number, etab, name, lab_lines, eq_lines, type_lines, top):
    """Une ligne : le numéro/établissement/nom sont à ``top`` ; chaque cellule est centrée autour de ce point."""
    words = [w(str(number), X["#"], top), w(etab, X["etab"], top), w(name, X["nom"], top)]
    for col, lines in (("lab", lab_lines), ("eq", eq_lines), ("type", type_lines)):
        offset = -(len(lines) - 1) / 2 * 16.4
        for i, line in enumerate(lines):
            x = X[col]
            for token in line.split():
                words.append(w(token, x, top + offset + i * 16.4))
                x += 6 * len(token) + 4
    return words


LAB = ["Laboratoire de Traitement", "de l'Information"]
TYPE = ["Membre", "Permanent(e)"]
SIDEBAR = [w("Accueil", 47.5, 162.0), w("", 51.7, 186.2), w("Propositions", 38.7, 201.0), w("MOHAMM", 40.0, 120.0)]
FOOTER = [w("Exporter", 138.0, 850.0), w("Records", 100.0, 900.0), w(":", 135.0, 900.0), w("3", 139.0, 900.0)]


def build_page():
    words = header() + SIDEBAR
    words += row(1, "FSBM", "ABDELHAK.CHAKLI", LAB, ["Télécommunications et Systèmes", "Automatiques"], TYPE, 173.6)
    words += row(2, "FSJESAS", "AHMED.EDDAOUI", LAB, ["Intelligence artificielle et Géo-", "informatique appliquées"], TYPE, 214.4)
    words += row(3, "FSBM", "Abdelmoghit.ZAARANE", LAB, ["Matériaux, Energies", "Renouvelables et", "Smart Technologies"], TYPE, 262.0)
    return words + FOOTER


def test_words_to_records_reads_multiline_cells_and_ignores_sidebar_and_footer():
    report = ExtractionReport(pdf="t.pdf")
    records = words_to_records(build_page(), 1, report)
    assert [r["row_number"] for r in records] == [1, 2, 3]
    first, second, third = records
    assert first["etablissement"] == "FSBM" and first["chercheur_source_name"] == "ABDELHAK.CHAKLI"
    assert first["laboratoire"] == "Laboratoire de Traitement de l'Information"
    assert first["equipe"] == "Télécommunications et Systèmes Automatiques" and first["type_membre"] == "Membre Permanent(e)"
    assert second["etablissement"] == "FSJESAS"
    assert second["equipe"] == "Intelligence artificielle et Géoinformatique appliquées"       # césure « Géo- » recollée
    assert third["equipe"] == "Matériaux, Energies Renouvelables et Smart Technologies"        # 3 lignes
    assert third["type_membre"] == "Membre Permanent(e)"                                       # pied de page non aspiré
    assert "MOHAMM" not in str(records) and "Exporter" not in str(records)
    assert report.header_found and not report.skipped


def test_layout_without_header_returns_nothing_so_other_strategies_can_run():
    assert words_to_records(SIDEBAR + FOOTER, 1, ExtractionReport(pdf="t.pdf")) == []


def test_join_wrapped_lines():
    assert join_wrapped_lines(["Géo-", "informatique appliquées"]) == "Géoinformatique appliquées"
    assert join_wrapped_lines(["Laboratoire de", "Chimie"]) == "Laboratoire de Chimie"
    assert join_wrapped_lines(["Système -", "Intelligent"]) == "Système - Intelligent"        # pas de césure : suite en majuscule
    assert join_wrapped_lines([]) == ""


def test_row_sequence_integrity_check():
    ok, bad = ExtractionReport(pdf="t"), ExtractionReport(pdf="t", expected_rows=5)
    check_row_sequence([{"row_number": 1}, {"row_number": 2}], ok)
    assert ok.warnings == []
    check_row_sequence([{"row_number": 1}, {"row_number": 3}], bad)
    assert len(bad.warnings) == 2                                                             # trou dans « # » + total ≠ annoncé


def test_page_records_feed_the_members_dataframe():
    records = words_to_records(build_page(), 1, ExtractionReport(pdf="t.pdf"))
    df = build_members_dataframe(records)
    assert df["chercheur_id"].tolist() == ["fsbm_abdelhak_chakli", "fsjesas_ahmed_eddaoui", "fsbm_abdelmoghit_zaarane"]
    assert df["nom_complet"].tolist()[2] == "Abdelmoghit Zaarane" and df["row_number"].tolist() == [1, 2, 3]
    stats = summarize_members(df, "FSBM")
    assert stats["membres_fsbm_uniques"] == 2 and stats["personnes_uniques_total"] == 3


# ---------------------------------------------------------------- vrai PDF (si présent)
REAL_PDF = Path(__file__).resolve().parents[1] / "data" / "input" / "Membres FSBM.pdf"


@pytest.mark.skipif(not REAL_PDF.exists(), reason="data/input/Membres FSBM.pdf absent (fichier non versionné)")
def test_real_pdf_extraction_matches_the_document():
    df, report = run_extraction(REAL_PDF)
    assert report.method == "word-layout" and report.expected_rows == 215 and report.warnings == []
    assert len(df) == 215 and df["chercheur_id"].is_unique and sorted(df["row_number"]) == list(range(1, 216))
    stats = summarize_members(df, "FSBM")
    assert stats["membres_fsbm_uniques"] == 207 and stats["laboratoires_fsbm"] == 11
    assert stats["etablissements"] == {"FSBM": 207, "FSJESAS": 3, "FLSHBM": 2, "ENCG": 2, "FMPC": 1}
    by_row = df.set_index("row_number")
    assert by_row.loc[1, "chercheur_id"] == "fsbm_abdelhak_chakli"
    assert by_row.loc[14, "equipe"] == "Intelligence artificielle et Géoinformatique appliquées"
    assert by_row.loc[121, "nom_complet"] == "Driss Bouggar" and by_row.loc[121, "chercheur_source_name"] == "DRISS.BOUGGAR"
    assert by_row.loc[202, "laboratoire"] == "Technologie de l'Information et Modélisation"
    assert by_row.loc[215, "chercheur_id"] == "fsbm_yman_chemlal"
    assert set(df["type_membre"]) == {"Membre Permanent(e)"}
