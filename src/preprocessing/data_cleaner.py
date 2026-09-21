"""Nettoyage, normalisation, déduplication et validation des données collectées.

Tables logiques produites : ``researchers`` / ``publications`` / ``researcher_publications``.
Règle d'or : aucune valeur n'est inventée ; une valeur invalide devient ``None`` et est comptée
dans le rapport qualité.
"""
from __future__ import annotations

import calendar
import logging
import re
from datetime import date
from difflib import SequenceMatcher
from typing import Any, Optional
from urllib.parse import urlparse

import pandas as pd
from pydantic import ValidationError

from src.data.extract_fsbm_members import is_target_institution
from src.data.schemas import (DOI_RE, Publication, ResearcherProfile, ScholarMetrics, assert_unique, year_max)
from src.preprocessing.text_cleaner import (build_embedding_text, clean_abstract_for_nlp, clean_text_light,
                                            extract_doi, looks_like_affiliation, normalize_doi, normalize_for_matching)

logger = logging.getLogger(__name__)

_MONTHS = {name.lower(): i for i, name in enumerate(calendar.month_name) if name}
_MONTHS |= {name.lower(): i for i, name in enumerate(calendar.month_abbr) if name}
_MONTHS["sept"] = 9


# --------------------------------------------------------------------------- petits utilitaires
def make_article_id(index: int) -> str:
    """Identifiant d'article séquentiel : 1 → ``art_00001``."""
    return f"art_{index:05d}"


def sanitize_count(value: Any) -> Optional[int]:
    """Entier ≥ 0 ou ``None`` (négatif, non numérique ou manquant → ``None``, jamais 0 par défaut)."""
    if value is None or (isinstance(value, float) and value != value):
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number >= 0 else None


def sanitize_url(value: Optional[str]) -> Optional[str]:
    """URL http(s) valide ou ``None``."""
    value = clean_text_light(value)
    if not value:
        return None
    parsed = urlparse(value)
    return value if parsed.scheme in {"http", "https"} and parsed.netloc else None


def sanitize_year(value: Any, year_min: int = 1950) -> Optional[int]:
    """Année plausible ∈ [year_min, année courante + 1] ou ``None``."""
    number = sanitize_count(value)
    return number if number is not None and year_min <= number <= year_max() else None


def parse_publication_date(raw: Optional[str], fallback_year: Optional[int] = None) -> tuple[Optional[str], Optional[int], Optional[str]]:
    """Analyse une date Scholar (``2021/1/1``, ``2021/3``, ``2021``, ``Jan 2021``, ``5 March 2021``…).

    Returns:
        ``(date_iso, année, précision)`` ; ``date_iso`` est ISO 8601 à la précision connue
        (``2021-03-05`` | ``2021-03`` | ``2021``). Si seule ``fallback_year`` est connue, précision « year ».
    """
    text = clean_text_light(raw) if raw else None
    year = month = day = None
    if text:
        if m := re.fullmatch(r"(\d{4})[/\-.](\d{1,2})[/\-.](\d{1,2})", text):
            year, month, day = (int(g) for g in m.groups())
        elif m := re.fullmatch(r"(\d{4})[/\-.](\d{1,2})", text):
            year, month = int(m.group(1)), int(m.group(2))
        elif m := re.fullmatch(r"(\d{4})", text):
            year = int(m.group(1))
        elif m := re.fullmatch(r"([A-Za-z]+)\.?\s+(?:(\d{1,2}),?\s+)?(\d{4})", text):
            month, day, year = _MONTHS.get(m.group(1).lower()), (int(m.group(2)) if m.group(2) else None), int(m.group(3))
        elif m := re.fullmatch(r"(\d{1,2})\s+([A-Za-z]+)\.?,?\s+(\d{4})", text):
            day, month, year = int(m.group(1)), _MONTHS.get(m.group(2).lower()), int(m.group(3))
    if year is None:
        year = sanitize_year(fallback_year)
        return (str(year), year, "year") if year else (None, None, None)
    if month is not None and not 1 <= month <= 12:
        month = day = None
    if day is not None:
        try:
            date(year, month or 1, day)
        except ValueError:
            day = None
    if month is None:
        return str(year), year, "year"
    if day is None:
        return f"{year:04d}-{month:02d}", year, "month"
    return f"{year:04d}-{month:02d}-{day:02d}", year, "day"


# --------------------------------------------------------------------------- publications : nettoyage
def clean_publication_record(raw: dict[str, Any], min_abstract_chars: int = 50, year_min: int = 1950) -> dict[str, Any]:
    """Nettoie une publication brute (le texte brut de l'abstract est conservé dans ``abstract``)."""
    titre = clean_text_light(raw.get("titre"))
    authors = [a for a in (clean_text_light(a) for a in raw.get("auteurs") or []) if a and a not in {"...", "…"}]
    date_iso, year, precision = parse_publication_date(raw.get("date_publication_raw"), sanitize_year(raw.get("annee"), year_min))
    doi = normalize_doi(raw.get("doi")) or extract_doi(raw.get("external_url"))

    abstract_raw = raw.get("abstract")
    if looks_like_affiliation(abstract_raw):          # contenu invalide (affiliations d'auteurs) : jamais utilisé comme abstract
        abstract_raw = None
    abstract_clean = clean_abstract_for_nlp(abstract_raw)
    status = raw.get("abstract_status") or "not_found"
    if abstract_clean is None:
        status = status if status in {"api_error", "publisher_unavailable"} else "not_found"
        abstract_raw = None
    elif len(abstract_clean) < min_abstract_chars:
        status = "too_short"
    elif status not in {"found", "truncated"}:
        status = "found"

    embedding_text, embedding_source = build_embedding_text(titre, abstract_clean, min_abstract_chars)
    return {
        "raw_pub_id": raw.get("raw_pub_id"), "chercheur_id": raw.get("chercheur_id"), "titre": titre,
        "auteurs": authors, "annee": year, "date_publication": date_iso, "date_precision": precision,
        "journal": clean_text_light(raw.get("journal")), "conference": clean_text_light(raw.get("conference")),
        "volume": clean_text_light(raw.get("volume")), "numero": clean_text_light(raw.get("numero")),
        "pages": clean_text_light(raw.get("pages")), "publisher": clean_text_light(raw.get("publisher")),
        "citations": sanitize_count(raw.get("citations")), "scholar_url": sanitize_url(raw.get("scholar_url")),
        "pdf_url": sanitize_url(raw.get("pdf_url")), "doi": doi if doi and DOI_RE.match(doi) else None,
        "abstract": abstract_raw, "abstract_clean": abstract_clean,
        "abstract_source": raw.get("abstract_source") if abstract_clean else None, "abstract_status": status,
        "embedding_text": embedding_text, "embedding_source": embedding_source,
        "scrape_status": raw.get("scrape_status"),
    }


# --------------------------------------------------------------------------- publications : déduplication
class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def _years_compatible(a: Optional[int], b: Optional[int], tolerance: int = 0) -> bool:
    return a is None or b is None or abs(a - b) <= tolerance


_ROMAN = {"i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x"}


def _marker_tokens(tokens: set[str]) -> set[str]:
    """Numéros de partie/volume (chiffres, chiffres romains) : « Part I » et « Part II » sont deux articles."""
    return {t for t in tokens if t.isdigit() or t in _ROMAN}


def _dois_conflict(a: dict[str, Any], b: dict[str, Any]) -> bool:
    return bool(a.get("doi") and b.get("doi") and a["doi"] != b["doi"])


def find_duplicate_groups(records: list[dict[str, Any]], fuzzy_threshold: float = 0.93) -> list[list[int]]:
    """Regroupe les indices de publications identiques.

    Priorité : (1) DOI normalisé ; (2) titre normalisé + année ; (3) flou prudent
    (même année ±1, similarité ≥ seuil, mêmes numéros de partie/volume). Deux DOI différents ne sont
    JAMAIS fusionnés.
    Les titres trop courts (< 3 mots) ne sont jamais fusionnés par le titre.
    """
    uf = _UnionFind(len(records))
    norm = [normalize_for_matching(r.get("titre")) for r in records]

    by_doi: dict[str, int] = {}
    for i, rec in enumerate(records):
        if rec.get("doi"):
            uf.union(i, by_doi.setdefault(rec["doi"], i))

    by_title: dict[str, list[int]] = {}
    for i, title in enumerate(norm):
        if len(title.split()) >= 3:
            by_title.setdefault(title, []).append(i)
    for indices in by_title.values():
        for pos, i in enumerate(indices):
            for j in indices[pos + 1:]:
                if _years_compatible(records[i].get("annee"), records[j].get("annee")) and not _dois_conflict(records[i], records[j]):
                    uf.union(i, j)

    tokens = [set(t.split()) for t in norm]
    by_year: dict[int, list[int]] = {}
    for i, rec in enumerate(records):
        if rec.get("annee") is not None and len(tokens[i]) >= 4:
            by_year.setdefault(rec["annee"], []).append(i)
    for year, indices in by_year.items():
        for pos, i in enumerate(indices):
            # même année (paires i<j) + année suivante (tolérance ±1 sans comparer deux fois)
            for j in indices[pos + 1:] + by_year.get(year + 1, []):
                if uf.find(i) == uf.find(j) or _dois_conflict(records[i], records[j]):
                    continue
                if len(tokens[i] & tokens[j]) / len(tokens[i] | tokens[j]) < 0.6:
                    continue
                if _marker_tokens(tokens[i]) != _marker_tokens(tokens[j]):
                    continue
                if SequenceMatcher(None, norm[i], norm[j]).ratio() >= fuzzy_threshold:
                    uf.union(i, j)

    groups: dict[int, list[int]] = {}
    for i in range(len(records)):
        groups.setdefault(uf.find(i), []).append(i)
    return list(groups.values())


def _record_quality(rec: dict[str, Any]) -> tuple:
    """Clé de tri : meilleur enregistrement = DOI > abstract complet > abstract long > citations."""
    return (bool(rec.get("doi")), rec.get("abstract_status") == "found", len(rec.get("abstract_clean") or ""),
            rec.get("citations") or 0)


def merge_group(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Fusionne les enregistrements d'un même article (complète les champs vides, sans rien inventer)."""
    ordered = sorted(records, key=_record_quality, reverse=True)
    merged = dict(ordered[0])
    for other in ordered[1:]:
        for key, value in other.items():
            if key in {"chercheur_id", "raw_pub_id"}:
                continue
            if merged.get(key) in (None, "", []) and value not in (None, "", []):
                merged[key] = value
    merged["auteurs"] = max((r.get("auteurs") or [] for r in records), key=len)
    cites = [r["citations"] for r in records if r.get("citations") is not None]
    merged["citations"] = max(cites) if cites else None
    merged["chercheur_ids"] = sorted({r["chercheur_id"] for r in records if r.get("chercheur_id")})
    merged["source_raw_pub_ids"] = [r["raw_pub_id"] for r in records if r.get("raw_pub_id")]
    merged["n_source_records"] = len(records)
    return merged


def deduplicate_publications(records: list[dict[str, Any]], fuzzy_threshold: float = 0.93) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Retourne ``(articles uniques avec article_id, liens chercheur↔article)``."""
    groups = find_duplicate_groups(records, fuzzy_threshold)
    merged = [merge_group([records[i] for i in group]) for group in groups]
    merged.sort(key=lambda a: (normalize_for_matching(a.get("titre")), a.get("annee") or 0))
    links: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for idx, article in enumerate(merged, start=1):
        article["article_id"] = make_article_id(idx)
        for raw_id in article["source_raw_pub_ids"]:
            key = (raw_id.split("::")[0], article["article_id"])  # chercheur_id ne contient jamais "::"
            if key not in seen:
                seen.add(key)
                links.append({"chercheur_id": key[0], "article_id": key[1], "raw_pub_id": raw_id})
    return merged, links


# --------------------------------------------------------------------------- chercheurs
def _metrics_from_profile(profile: Optional[dict[str, Any]]) -> dict[str, Any]:
    """Métriques Scholar ; ``*_since_2021`` n'est renseigné que si la fenêtre Scholar est bien « depuis 2021 »."""
    metrics = (profile or {}).get("metrics") or {}
    since_year = (profile or {}).get("since_year")
    out = {
        "citations_totales": sanitize_count(metrics.get("citations")),
        "h_index": sanitize_count(metrics.get("h_index")),
        "i10_index": sanitize_count(metrics.get("i10_index")),
        "since_year": since_year,
        "citations_since": sanitize_count(metrics.get("citations_since")),
        "h_index_since": sanitize_count(metrics.get("h_index_since")),
        "i10_index_since": sanitize_count(metrics.get("i10_index_since")),
    }
    is_2021 = since_year == 2021
    out["citations_since_2021"] = out["citations_since"] if is_2021 else None
    out["h_index_since_2021"] = out["h_index_since"] if is_2021 else None
    out["i10_index_since_2021"] = out["i10_index_since"] if is_2021 else None
    return out


RESEARCHER_COLUMNS = [
    "chercheur_id", "nom_complet", "chercheur_source_name", "etablissement", "laboratoire", "equipe", "type_membre",
    "n_affectations", "scholar_id", "scholar_url", "affiliation", "email_domain", "interests", "citations_totales",
    "h_index", "i10_index", "since_year", "citations_since", "h_index_since", "i10_index_since", "citations_since_2021",
    "h_index_since_2021", "i10_index_since_2021", "scholar_profile_status", "profile_match_confidence", "match_method",
    "collection_status", "status_detail"]


def _join_unique(values: pd.Series, sep: str = " | ") -> Optional[str]:
    items = list(dict.fromkeys(v for v in values.dropna().astype(str) if v.strip()))
    return sep.join(items) if items else None


def _nn(value: Any) -> Any:
    """NaN/NA pandas → ``None`` (une absence reste une absence, jamais une valeur numérique ou une chaîne « nan »)."""
    return None if value is None or (not isinstance(value, (list, str)) and pd.isna(value)) else value


def clean_researchers(members: pd.DataFrame, scholars_raw: list[dict[str, Any]], institution: str = "FSBM",
                      restrict_to: Optional[set[str]] = None,
                      known_scholar_ids: Optional[dict[str, str]] = None) -> pd.DataFrame:
    """Une ligne par chercheur de l'établissement cible, fusionnée avec son état Scholar.

    ``restrict_to`` : si fourni, ne garde que ces ``chercheur_id`` (ex. ceux qui ont un Scholar ID).
    ``known_scholar_ids`` : ``chercheur_id → Scholar ID`` saisis à la main, utilisés tant que le profil n'est pas collecté.
    """
    members = members[is_target_institution(members["etablissement"], institution)]
    members = members[~members["is_duplicate_row"].astype(bool)]
    if restrict_to is not None:
        members = members[members["chercheur_id"].isin(restrict_to)]
    grouped = members.groupby("chercheur_id", sort=False).agg(
        nom_complet=("nom_complet", "first"), chercheur_source_name=("chercheur_source_name", "first"),
        etablissement=("etablissement", "first"), laboratoire=("laboratoire", _join_unique),
        equipe=("equipe", _join_unique), type_membre=("type_membre", _join_unique),
        n_affectations=("chercheur_id", "size")).reset_index()

    states = {s["chercheur_id"]: s for s in scholars_raw}
    rows = []
    for rec in grouped.to_dict("records"):
        state = states.get(rec["chercheur_id"], {})
        profile = state.get("profile")
        scholar_id = state.get("scholar_id") or (known_scholar_ids or {}).get(rec["chercheur_id"]) or None
        scholar_url = sanitize_url(state.get("scholar_url")) or (
            f"https://scholar.google.com/citations?user={scholar_id}" if scholar_id else None)
        rows.append({**rec,
                     "scholar_id": scholar_id, "scholar_url": scholar_url,
                     "affiliation": clean_text_light((profile or {}).get("affiliation")),
                     "email_domain": (profile or {}).get("email_domain"),
                     "interests": [clean_text_light(i) for i in (profile or {}).get("interests") or [] if clean_text_light(i)],
                     **_metrics_from_profile(profile),
                     "scholar_profile_status": state.get("scholar_profile_status", "pending"),
                     "profile_match_confidence": state.get("profile_match_confidence"),
                     "match_method": state.get("match_method"),
                     "collection_status": state.get("collection_status", "n/a"),
                     "status_detail": state.get("status_detail")})
    df = pd.DataFrame(rows, columns=RESEARCHER_COLUMNS)      # colonnes garanties même sans aucun chercheur
    for col in ("citations_totales", "h_index", "i10_index", "citations_since", "h_index_since", "i10_index_since",
                "citations_since_2021", "h_index_since_2021", "i10_index_since_2021", "since_year"):
        df[col] = pd.array(df[col], dtype="Int64")
    return df


# --------------------------------------------------------------------------- pipeline
def clean_publications(pubs_raw: list[dict[str, Any]], min_abstract_chars: int = 50, year_min: int = 1950,
                       fuzzy_threshold: float = 0.93, valid_researchers: Optional[set[str]] = None) -> dict[str, Any]:
    """Nettoie + déduplique + valide. Retourne articles, liens, statistiques de nettoyage et erreurs."""
    cleaned, dropped_no_title, outside = [], 0, 0
    for raw in pubs_raw:
        if valid_researchers is not None and raw.get("chercheur_id") not in valid_researchers:
            outside += 1
            continue
        rec = clean_publication_record(raw, min_abstract_chars, year_min)
        if not rec["titre"]:
            dropped_no_title += 1
            continue
        cleaned.append(rec)

    articles, links = deduplicate_publications(cleaned, fuzzy_threshold)
    validation_errors = validate_articles(articles)
    assert_unique([a["article_id"] for a in articles], "article_id")
    stats = {"raw_records": len(pubs_raw), "records_without_title_dropped": dropped_no_title,
             "records_outside_target_institution": outside, "records_after_cleaning": len(cleaned),
             "unique_articles": len(articles),
             "duplicates_merged": len(cleaned) - len(articles),
             "links": len(links)}
    return {"articles": articles, "links": links, "stats": stats, "validation_errors": validation_errors}


def validate_articles(articles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Valide chaque article avec Pydantic ; retourne la liste des erreurs (article_id, champ, message)."""
    errors: list[dict[str, Any]] = []
    fields = set(Publication.model_fields)
    for art in articles:
        try:
            Publication(**{k: v for k, v in art.items() if k in fields})
        except ValidationError as exc:
            for err in exc.errors():
                errors.append({"article_id": art.get("article_id"), "field": ".".join(map(str, err["loc"])),
                               "message": err["msg"]})
    return errors


def validate_researchers(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Valide les chercheurs (métriques ≥ 0, statut connu, confiance ∈ [0,1], URL valide)."""
    errors: list[dict[str, Any]] = []
    for row in df.to_dict("records"):
        metrics = {k: (None if pd.isna(row[k]) else int(row[k])) for k in ("citations_totales", "h_index", "i10_index")}
        conf = row.get("profile_match_confidence")
        try:
            ResearcherProfile(
                chercheur_id=row["chercheur_id"], nom_complet=row["nom_complet"],
                scholar_id=_nn(row.get("scholar_id")) or None, scholar_url=_nn(row.get("scholar_url")) or None,
                interests=_nn(row.get("interests")) or [], metriques=ScholarMetrics(**metrics),
                scholar_profile_status=row["scholar_profile_status"],
                profile_match_confidence=None if conf is None or pd.isna(conf) else float(conf))
        except ValidationError as exc:
            for err in exc.errors():
                errors.append({"chercheur_id": row["chercheur_id"], "field": ".".join(map(str, err["loc"])),
                               "message": err["msg"]})
    assert_unique(df["chercheur_id"].tolist(), "chercheur_id")
    return errors


def articles_to_dataframe(articles: list[dict[str, Any]]) -> pd.DataFrame:
    """DataFrame des publications (colonnes stables ; listes conservées pour le Parquet)."""
    cols = ["article_id", "titre", "auteurs", "annee", "date_publication", "date_precision", "journal", "conference",
            "volume", "numero", "pages", "publisher", "citations", "scholar_url", "pdf_url", "doi", "abstract",
            "abstract_clean", "abstract_source", "abstract_status", "embedding_text", "embedding_source",
            "scrape_status", "chercheur_ids", "n_source_records"]
    df = pd.DataFrame([{c: a.get(c) for c in cols} for a in articles], columns=cols)
    df["annee"] = pd.array(df["annee"], dtype="Int64")
    df["citations"] = pd.array(df["citations"], dtype="Int64")
    df["n_chercheurs_fsbm"] = df["chercheur_ids"].map(len)
    return df


def build_dataset_final(researchers: pd.DataFrame, articles_df: pd.DataFrame, links: pd.DataFrame) -> list[dict[str, Any]]:
    """Structure JSON demandée (chercheur → métriques → articles), embeddings ajoutés plus tard."""
    art_by_id = {r["article_id"]: r for r in articles_df.to_dict("records")}
    ids_by_researcher = links.groupby("chercheur_id")["article_id"].apply(list).to_dict()
    dataset = []
    for row in researchers.to_dict("records"):
        if row["scholar_profile_status"] != "matched":
            continue
        articles = []
        for art_id in ids_by_researcher.get(row["chercheur_id"], []):
            a = art_by_id[art_id]
            articles.append({
                "article_id": art_id, "titre": a["titre"], "auteurs": list(a["auteurs"]),
                "annee": a["annee"], "date_publication": a["date_publication"], "journal": a["journal"],
                "conference": a["conference"], "citations": a["citations"], "doi": a["doi"],
                "scholar_url": a["scholar_url"], "abstract": a["abstract"], "abstract_clean": a["abstract_clean"],
                "abstract_status": a["abstract_status"], "embedding_source": a["embedding_source"],
                "embedding_zembed1": None})
        dataset.append({
            # schéma du sujet : « chercheur_id » = identifiant Scholar unique ; l'identifiant interne du PDF est conservé
            "chercheur_id": row["scholar_id"] or row["chercheur_id"], "chercheur_id_interne": row["chercheur_id"],
            "scholar_id": row["scholar_id"], "nom_complet": row["nom_complet"],
            "affiliation": row["affiliation"], "laboratoire": row["laboratoire"], "equipe": row["equipe"],
            "interests": list(row["interests"]), "scholar_profile_status": row["scholar_profile_status"],
            "profile_match_confidence": row["profile_match_confidence"],
            "metriques": {k: row[k] for k in ("citations_totales", "h_index", "i10_index",
                                              "citations_since_2021", "h_index_since_2021", "i10_index_since_2021")},
            "articles": articles})
    return dataset


def compute_quality_report(researchers: pd.DataFrame, articles_df: pd.DataFrame, links: pd.DataFrame,
                           pubs_raw: list[dict[str, Any]], scholars_raw: list[dict[str, Any]],
                           clean_stats: dict[str, Any], validation_errors: list[dict[str, Any]]) -> dict[str, Any]:
    """Rapport qualité des données (pourcentages calculés sur le dataset réellement collecté)."""
    n_art, n_res = len(articles_df), len(researchers)
    pct = lambda num, den: round(100 * num / den, 1) if den else None  # noqa: E731
    with_abstract = int(articles_df["abstract_clean"].notna().sum())
    detail_errors = sum(1 for p in pubs_raw if p.get("scrape_status") == "detail_error")
    researcher_errors = sum(1 for s in scholars_raw if s.get("scholar_profile_status") in {"error", "blocked"}
                            or s.get("collection_status") in {"error", "blocked"})
    return {
        "chercheurs_fsbm_dans_pdf": clean_stats.get("researchers_in_pdf"),
        "chercheurs_avec_scholar_id": clean_stats.get("researchers_with_scholar_id"),
        "chercheurs_fsbm": n_res,
        "chercheurs_par_statut_scholar": researchers["scholar_profile_status"].value_counts().to_dict(),
        "pct_chercheurs_sans_scholar": pct(int((researchers["scholar_profile_status"] != "matched").sum()), n_res),
        "publications_uniques": n_art,
        "liens_chercheur_publication": len(links),
        "doublons_fusionnes": clean_stats["duplicates_merged"],
        "publications_avec_doi": int(articles_df["doi"].notna().sum()),
        "pct_publications_avec_doi": pct(int(articles_df["doi"].notna().sum()), n_art),
        "abstracts_recuperes": with_abstract,
        "pct_publications_sans_abstract": pct(n_art - with_abstract, n_art),
        "abstracts_par_statut": articles_df["abstract_status"].value_counts().to_dict(),
        "abstracts_par_source": articles_df["abstract_source"].fillna("(aucune)").value_counts().to_dict(),
        "embedding_source_prevue": articles_df["embedding_source"].value_counts().to_dict(),
        "publications_sans_annee": int(articles_df["annee"].isna().sum()),
        "erreurs_scraping": {"details_publications_en_erreur": detail_errors,
                             "chercheurs_en_erreur_ou_bloques": researcher_errors},
        "erreurs_validation": len(validation_errors),
        "nettoyage": clean_stats,
    }


def quality_report_markdown(report: dict[str, Any]) -> str:
    """Version lisible (Markdown) du rapport qualité."""
    lines = ["# Rapport qualité des données", "",
             "_Les pourcentages décrivent uniquement le dataset collecté, pas la production scientifique totale de la FSBM._", ""]
    for key, value in report.items():
        if isinstance(value, dict):
            lines.append(f"## {key}")
            lines.extend(f"- **{k}** : {v}" for k, v in value.items())
            lines.append("")
        else:
            lines.append(f"- **{key}** : {value}")
    return "\n".join(lines) + "\n"
