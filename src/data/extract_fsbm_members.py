"""Étape 0 — Extraction de la liste des chercheurs depuis « Membres FSBM.pdf ».

Le PDF est la SOURCE DE RÉFÉRENCE : aucun nom n'est inventé. Colonnes attendues :
Etablissement | Enseignant Chercheur | Laboratoire | Equipe | Type Membre.

Stratégie d'extraction (par ordre) :
  1. mise en page par coordonnées de mots (impression d'une page web : cellules multi-lignes centrées
     verticalement, colonnes déduites de la ligne d'en-tête, ligne rattachée au numéro « # » le plus proche) ;
  2. tableaux à bordures (pdfplumber, stratégie « lines ») ;
  3. tableaux sans bordures (stratégie « text ») ;
  4. repli sur les lignes de texte découpées sur >= 2 espaces (les lignes non
     analysables sont listées dans le rapport pour vérification manuelle).

Contrôle d'intégrité : les numéros « # » doivent former la suite 1…N et N doit correspondre au total
affiché en pied de page (« Records : N sur N ») ; sinon des avertissements sont émis.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

import pandas as pd

from src.preprocessing.text_cleaner import clean_text_light, normalize_for_matching, slugify

logger = logging.getLogger(__name__)

RAW_COLUMNS = ["chercheur_id", "chercheur_source_name", "nom_complet", "etablissement",
               "laboratoire", "equipe", "type_membre", "is_duplicate_row", "row_number", "source_page"]

# alias d'en-têtes (forme normalisée) -> champ interne
HEADER_ALIASES: dict[str, tuple[str, ...]] = {
    "etablissement": ("etablissement", "etab", "institution"),
    "chercheur_source_name": ("enseignant chercheur", "enseignant", "chercheur", "nom et prenom",
                              "nom prenom", "nom", "membre"),
    "laboratoire": ("laboratoire", "labo", "structure de recherche"),
    "equipe": ("equipe", "equipe de recherche"),
    "type_membre": ("type membre", "type de membre", "type", "statut"),
}
_LEADING_TITLES = {"pr", "prof", "professeur", "dr", "docteur", "mme", "mlle", "mr"}


# --------------------------------------------------------------------------- noms
def _capitalize_token(token: str) -> str:
    """``m'sik`` → ``M'Sik`` ; ``el-amrani`` → ``El-Amrani``."""
    parts = re.split(r"([-'’])", token.lower())
    return "".join(p if p in "-'’" else p.capitalize() for p in parts)


def normalize_person_name(raw: Optional[str]) -> str:
    """Normalise un nom brut du PDF : ``DRISS.BOUGGAR`` → ``Driss Bouggar``.

    - les points et underscores servent de séparateurs ;
    - les titres initiaux (Pr, Dr, Mme…) sont retirés ;
    - chaque mot est mis en casse « Titre » (traits d'union et apostrophes gérés).

    L'ordre nom/prénom du PDF est conservé (il n'est pas déductible de façon fiable) ;
    les comparaisons ultérieures sont insensibles à l'ordre (voir ``name_key``).
    """
    text = clean_text_light(raw) or ""
    text = re.sub(r"[._]+", " ", text)
    tokens = [t for t in text.split() if t]
    while len(tokens) > 1 and tokens[0].lower() in _LEADING_TITLES:
        tokens.pop(0)
    return " ".join(_capitalize_token(t) for t in tokens)


def name_key(nom_complet: str) -> str:
    """Clé insensible à l'ordre, aux accents et à la casse (``Bouggar Driss`` == ``Driss Bouggar``)."""
    return " ".join(sorted(normalize_for_matching(nom_complet).split()))


def make_researcher_id(nom_complet: str, etablissement: Optional[str] = "FSBM") -> str:
    """Identifiant interne stable, ex. ``fsbm_driss_bouggar``. Ce n'est PAS l'identifiant Scholar."""
    prefix = slugify(etablissement or "") or "unk"
    slug = slugify(nom_complet) or "unknown"
    return f"{prefix}_{slug}"


# --------------------------------------------------------------------------- lignes → enregistrements
def detect_header_mapping(row: Iterable[Optional[str]]) -> dict[int, str]:
    """Si ``row`` ressemble à un en-tête, retourne ``{index_colonne: champ}`` ; sinon ``{}``.

    Un en-tête valide doit reconnaître au moins la colonne « nom » et une autre colonne.
    """
    mapping: dict[int, str] = {}
    for idx, cell in enumerate(row):
        norm = normalize_for_matching(cell)
        if not norm:
            continue
        for field_name, aliases in HEADER_ALIASES.items():
            if field_name in mapping.values():
                continue
            if norm in aliases:
                mapping[idx] = field_name
                break
    if "chercheur_source_name" in mapping.values() and len(mapping) >= 2:
        return mapping
    return {}


@dataclass
class ExtractionReport:
    """Trace de ce qui a été lu dans le PDF (transparence sur le parsing)."""

    pdf: str
    method: str = ""
    pages: int = 0
    rows_seen: int = 0
    rows_kept: int = 0
    header_found: bool = False
    expected_rows: Optional[int] = None          # total annoncé en pied de page (« Records : N sur N »)
    skipped: list[dict[str, Any]] = field(default_factory=list)
    unparsed_lines: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def rows_to_records(rows: list[list[Optional[str]]], page: int, mapping: dict[int, str],
                    report: ExtractionReport) -> tuple[list[dict[str, Any]], dict[int, str]]:
    """Convertit des lignes de tableau en enregistrements bruts.

    ``mapping`` est propagé d'une page à l'autre (les en-têtes ne sont pas répétés partout).
    """
    records: list[dict[str, Any]] = []
    for row in rows:
        report.rows_seen += 1
        header = detect_header_mapping(row)
        if header:
            mapping = header
            report.header_found = True
            continue
        if not mapping:
            report.skipped.append({"page": page, "reason": "no_header_yet", "row": row})
            continue
        rec: dict[str, Any] = {f: None for f in HEADER_ALIASES}
        for idx, fname in mapping.items():
            if idx < len(row):
                rec[fname] = clean_text_light(row[idx])
        if not rec["chercheur_source_name"]:
            report.skipped.append({"page": page, "reason": "empty_name", "row": row})
            continue
        rec["source_page"] = page
        records.append(rec)
    return records, mapping


def _split_text_line(line: str) -> list[str]:
    return [c.strip() for c in re.split(r"\s{2,}|\t", line.strip()) if c.strip()]


# --------------------------------------------------------------------------- mise en page par coordonnées
COL_TOL = 3.0                # tolérance (pt) sur le bord gauche d'une colonne
LINE_TOL = 3.0               # écart vertical max (pt) entre deux mots d'une même ligne
MAX_ROW_HALF_HEIGHT = 45.0   # distance verticale max (pt) entre un mot et le numéro de sa ligne
_HEADER_WORDS = {"etablissement": "etablissement", "enseignant": "chercheur_source_name",
                 "laboratoire": "laboratoire", "equipe": "equipe", "type": "type_membre"}
_FOOTER_WORDS = {"exporter", "records"}
_ICON_RE = re.compile("^[-]+$")          # glyphes d'icônes (police d'interface) : ignorés
_TOTAL_RE = re.compile(r"Records\s*:\s*\d+\s*sur\s*(\d+)")


def _find_header(words: list[dict[str, Any]]) -> Optional[tuple[float, list[tuple[float, str]]]]:
    """Repère la ligne d'en-tête : retourne ``(bas de la ligne, [(x0, champ)…] triés)`` ou ``None``."""
    for anchor in words:
        if normalize_for_matching(anchor["text"]) != "etablissement":
            continue
        line = [w for w in words if abs(w["top"] - anchor["top"]) < LINE_TOL]
        cols: dict[str, float] = {}
        for w in line:
            key = "index" if w["text"] == "#" else _HEADER_WORDS.get(normalize_for_matching(w["text"]))
            if key and key not in cols:
                cols[key] = w["x0"]
        if {"index", "etablissement", "chercheur_source_name"} <= cols.keys():
            return max(w["bottom"] for w in line), sorted((x, f) for f, x in cols.items())
    return None


def _column_of(x0: float, cols: list[tuple[float, str]]) -> Optional[str]:
    field = None
    for x, name in cols:
        if x0 >= x - COL_TOL:
            field = name
    return field


def _group_lines(words: list[dict[str, Any]]) -> list[str]:
    """Regroupe les mots d'une cellule en lignes (tri vertical puis horizontal)."""
    lines: list[list[Any]] = []
    for w in sorted(words, key=lambda w: (w["top"], w["x0"])):
        center = (w["top"] + w["bottom"]) / 2
        if lines and abs(lines[-1][0] - center) < LINE_TOL:
            lines[-1][1].append(w)
        else:
            lines.append([center, [w]])
    return [" ".join(x["text"] for x in sorted(ws, key=lambda x: x["x0"])) for _, ws in lines]


def join_wrapped_lines(lines: list[str]) -> str:
    """Recolle les lignes d'une cellule ; une césure de fin de ligne (« Géo- » + « informatique ») est résorbée."""
    out = ""
    for line in lines:
        if not out:
            out = line
        elif out.endswith("-") and line[:1].islower():
            out = out[:-1] + line
        else:
            out += " " + line
    return out


def words_to_records(words: list[dict[str, Any]], page: int, report: ExtractionReport) -> list[dict[str, Any]]:
    """Convertit les mots (avec coordonnées) d'une page en enregistrements ; ``[]`` si la mise en page diffère."""
    words = [w for w in words if not _ICON_RE.match(w["text"])]
    header = _find_header(words)
    if header is None:
        return []
    header_bottom, cols = header
    report.header_found = True
    footer_top = min((w["top"] for w in words if w["top"] > header_bottom
                      and normalize_for_matching(w["text"]) in _FOOTER_WORDS), default=float("inf"))
    left = cols[0][0] - COL_TOL                                  # exclut la barre latérale
    body = [w for w in words if header_bottom - 1 < w["top"] < footer_top and w["x0"] >= left]
    anchors = [w for w in body if _column_of(w["x0"], cols) == "index" and w["text"].isdigit()]
    if not anchors:
        return []
    centers = [(a["top"] + a["bottom"]) / 2 for a in anchors]
    cells: list[dict[str, list[dict[str, Any]]]] = [{} for _ in anchors]
    for w in body:
        center = (w["top"] + w["bottom"]) / 2
        idx = min(range(len(anchors)), key=lambda k: abs(centers[k] - center))
        field = _column_of(w["x0"], cols)
        if field == "index":
            continue
        if field is None or abs(centers[idx] - center) > MAX_ROW_HALF_HEIGHT:
            report.skipped.append({"page": page, "reason": "word_outside_rows", "row": w["text"]})
            continue
        cells[idx].setdefault(field, []).append(w)

    records = []
    for anchor, row_cells in zip(anchors, cells):
        report.rows_seen += 1
        rec: dict[str, Any] = {f: None for f in HEADER_ALIASES}
        for field, ws in row_cells.items():
            rec[field] = clean_text_light(join_wrapped_lines(_group_lines(ws)))
        if not rec["chercheur_source_name"]:
            report.skipped.append({"page": page, "reason": "empty_name", "row": anchor["text"]})
            continue
        rec.update(row_number=int(anchor["text"]), source_page=page)
        records.append(rec)
    return records


def check_row_sequence(records: list[dict[str, Any]], report: ExtractionReport) -> None:
    """Contrôle d'intégrité : numéros 1…N sans trou ni doublon, et N = total annoncé en pied de page."""
    numbers = [r["row_number"] for r in records if r.get("row_number") is not None]
    if sorted(numbers) != list(range(1, len(numbers) + 1)):
        missing = sorted(set(range(1, max(numbers, default=0) + 1)) - set(numbers))
        report.warnings.append(f"Numérotation « # » incohérente (manquants : {missing[:10]}).")
    if report.expected_rows is not None and report.expected_rows != len(records):
        report.warnings.append(f"{len(records)} lignes extraites mais le PDF annonce {report.expected_rows} enregistrements.")


def _run_word_layout(pdf: Any, pdf_path: Path) -> tuple[list[dict[str, Any]], ExtractionReport]:
    report = ExtractionReport(pdf=pdf_path.name, method="word-layout", pages=len(pdf.pages))
    records: list[dict[str, Any]] = []
    for page_no, page in enumerate(pdf.pages, start=1):
        page_records = words_to_records(page.extract_words(), page_no, report)
        if not page_records:
            report.warnings.append(f"Page {page_no} : en-tête ou lignes numérotées introuvables (page ignorée).")
        records.extend(page_records)
        if match := _TOTAL_RE.search(page.extract_text() or ""):
            report.expected_rows = int(match.group(1))
    report.rows_kept = len(records)
    if records:
        check_row_sequence(records, report)
    return records, report


def extract_records_from_pdf(pdf_path: Path | str) -> tuple[list[dict[str, Any]], ExtractionReport]:
    """Extrait les enregistrements bruts (sans normalisation) du PDF."""
    import pdfplumber  # import local : dépendance lourde, inutile pour les tests de nettoyage

    pdf_path = Path(pdf_path)
    report = ExtractionReport(pdf=pdf_path.name)
    records: list[dict[str, Any]] = []
    with pdfplumber.open(pdf_path) as pdf:
        report.pages = len(pdf.pages)
        if not any(page.chars for page in pdf.pages):
            raise ValueError("Le PDF ne contient aucune couche texte (scan ?). Un OCR est nécessaire avant extraction.")
        records, report = _run_word_layout(pdf, pdf_path)
        if records:
            return records, report
        for method, settings in (("table-lines", None),
                                 ("table-text", {"vertical_strategy": "text", "horizontal_strategy": "text"})):
            records, report = _run_table_method(pdf, method, settings, pdf_path)
            if records:
                return records, report
        records, report = _run_text_fallback(pdf, pdf_path)
    return records, report


def _run_table_method(pdf: Any, method: str, settings: Optional[dict], pdf_path: Path) -> tuple[list[dict[str, Any]], ExtractionReport]:
    report = ExtractionReport(pdf=pdf_path.name, method=method, pages=len(pdf.pages))
    records: list[dict[str, Any]] = []
    mapping: dict[int, str] = {}
    for page_no, page in enumerate(pdf.pages, start=1):
        tables = page.extract_tables(settings) if settings else page.extract_tables()
        for table in tables:
            recs, mapping = rows_to_records(table, page_no, mapping, report)
            records.extend(recs)
    report.rows_kept = len(records)
    return records, report


def _run_text_fallback(pdf: Any, pdf_path: Path) -> tuple[list[dict[str, Any]], ExtractionReport]:
    report = ExtractionReport(pdf=pdf_path.name, method="text-lines", pages=len(pdf.pages))
    report.warnings.append("Aucun tableau détecté : repli sur le découpage de lignes (vérifier le résultat).")
    records: list[dict[str, Any]] = []
    mapping: dict[int, str] = {}
    for page_no, page in enumerate(pdf.pages, start=1):
        rows = [_split_text_line(line) for line in (page.extract_text() or "").splitlines() if line.strip()]
        for row in rows:
            header = detect_header_mapping(row)
            if header:
                mapping = header
                report.header_found = True
                continue
            if mapping and len(row) == len(mapping):
                recs, mapping = rows_to_records([row], page_no, mapping, report)
                records.extend(recs)
            else:
                report.unparsed_lines.append(f"p{page_no}: {' | '.join(row)}")
    report.rows_kept = len(records)
    return records, report


# --------------------------------------------------------------------------- DataFrame
def build_members_dataframe(records: list[dict[str, Any]], forward_fill: Iterable[str] = ()) -> pd.DataFrame:
    """Normalise les enregistrements bruts et calcule ``chercheur_id`` + doublons."""
    df = pd.DataFrame(records)
    if df.empty:
        return pd.DataFrame(columns=RAW_COLUMNS)
    for col in ("etablissement", "laboratoire", "equipe", "type_membre", "row_number", "source_page"):
        if col not in df:
            df[col] = None
    for col in forward_fill:
        if col in df:
            df[col] = df[col].ffill()

    df["nom_complet"] = df["chercheur_source_name"].map(normalize_person_name)
    df = df[df["nom_complet"] != ""].copy()

    # un même chercheur (clé insensible à l'ordre) garde un seul identifiant, même s'il a plusieurs lignes
    ids: dict[tuple[str, str], str] = {}
    out_ids = []
    for nom, etab in zip(df["nom_complet"], df["etablissement"]):
        key = (slugify(etab or ""), name_key(nom))
        ids.setdefault(key, make_researcher_id(nom, etab))
        out_ids.append(ids[key])
    df["chercheur_id"] = out_ids
    df["nom_complet"] = df.groupby("chercheur_id")["nom_complet"].transform("first")

    dup_cols = ["chercheur_id", "laboratoire", "equipe", "type_membre"]
    df["is_duplicate_row"] = df.duplicated(subset=dup_cols, keep="first")
    return df[RAW_COLUMNS].reset_index(drop=True)


def is_target_institution(series: pd.Series, institution: str) -> pd.Series:
    """Comparaison tolérante (casse/espaces) de la colonne Etablissement."""
    target = normalize_for_matching(institution)
    return series.fillna("").map(normalize_for_matching) == target


def unique_researchers(df: pd.DataFrame, institution: str = "FSBM") -> list[dict[str, Any]]:
    """Un enregistrement par chercheur de l'établissement cible (laboratoires/équipes multiples joints par « | »)."""
    subset = df[is_target_institution(df["etablissement"], institution) & ~df["is_duplicate_row"].astype(bool)]
    records: list[dict[str, Any]] = []
    for cid, group in subset.groupby("chercheur_id", sort=False):
        join = lambda col: " | ".join(dict.fromkeys(v for v in group[col].dropna().astype(str) if v.strip())) or None  # noqa: E731
        first = group.iloc[0]
        records.append({"chercheur_id": cid, "nom_complet": first["nom_complet"], "etablissement": first["etablissement"],
                        "laboratoire": join("laboratoire"), "equipe": join("equipe"), "type_membre": join("type_membre")})
    return records


def summarize_members(df: pd.DataFrame, institution: str = "FSBM") -> dict[str, Any]:
    """Statistiques demandées : total, FSBM, laboratoires, équipes, répartitions, doublons."""
    fsbm = df[is_target_institution(df["etablissement"], institution)]
    fsbm_unique = fsbm.drop_duplicates("chercheur_id")
    by_lab = fsbm_unique.groupby(fsbm_unique["laboratoire"].fillna("(non renseigné)"))["chercheur_id"].nunique()
    team_frame = fsbm.assign(_lab=fsbm["laboratoire"].fillna("(non renseigné)"),
                             _team=fsbm["equipe"].fillna("(non renseignée)"))
    by_team = team_frame.groupby(["_lab", "_team"])["chercheur_id"].nunique()
    memberships = df[~df["is_duplicate_row"].astype(bool)].groupby("chercheur_id").size()   # affectations distinctes
    return {
        "lignes_extraites": int(len(df)),
        "personnes_uniques_total": int(df["chercheur_id"].nunique()),
        "membres_fsbm_uniques": int(fsbm_unique["chercheur_id"].nunique()),
        "etablissements": df["etablissement"].fillna("(non renseigné)").value_counts().to_dict(),
        "laboratoires_fsbm": int(fsbm["laboratoire"].nunique()),
        "equipes_fsbm": int(team_frame[["_lab", "_team"]].drop_duplicates().shape[0]),
        "chercheurs_par_laboratoire": by_lab.sort_values(ascending=False).to_dict(),
        "chercheurs_par_equipe": {f"{lab} / {team}": int(n) for (lab, team), n in by_team.sort_values(ascending=False).items()},
        "lignes_doublons_exacts": int(df["is_duplicate_row"].sum()),
        "personnes_avec_plusieurs_affectations": int((memberships > 1).sum()),
        "valeurs_manquantes": {c: int(df[c].isna().sum()) for c in ("etablissement", "laboratoire", "equipe", "type_membre")},
        "types_membre": df["type_membre"].fillna("(non renseigné)").value_counts().to_dict(),
    }


def inspect_pdf(pdf_path: Path | str, max_pages: int = 2, max_rows: int = 8) -> str:
    """Aperçu brut de la mise en page (à utiliser si l'extraction automatique échoue)."""
    import pdfplumber

    lines: list[str] = []
    with pdfplumber.open(pdf_path) as pdf:
        lines.append(f"Pages : {len(pdf.pages)}")
        for i, page in enumerate(pdf.pages[:max_pages], start=1):
            lines.append(f"\n--- page {i} : {len(page.chars)} caractères ---")
            for t_idx, table in enumerate(page.extract_tables()):
                lines.append(f"[table {t_idx}] {len(table)} lignes, {max(len(r) for r in table)} colonnes")
                lines.extend(f"  {row}" for row in table[:max_rows])
            lines.append("[texte brut]")
            lines.extend("  " + ln for ln in (page.extract_text() or "").splitlines()[:max_rows])
    return "\n".join(lines)


def run_extraction(pdf_path: Path | str, forward_fill: Iterable[str] = ()) -> tuple[pd.DataFrame, ExtractionReport]:
    """Pipeline complet : PDF → DataFrame normalisé + rapport d'extraction."""
    records, report = extract_records_from_pdf(pdf_path)
    if not records:
        raise ValueError("Aucun chercheur extrait. Lancez avec --inspect pour voir la mise en page du PDF.")
    df = build_members_dataframe(records, forward_fill)
    report.rows_kept = len(df)
    logger.info("Extraction (%s) : %d lignes conservées sur %d lues", report.method, report.rows_kept, report.rows_seen)
    return df, report
