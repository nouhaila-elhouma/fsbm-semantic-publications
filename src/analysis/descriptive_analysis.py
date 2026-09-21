"""Analyses descriptives du dataset collecté (graphiques matplotlib + tables CSV).

ATTENTION : ces classements décrivent uniquement le dataset collecté (profils Scholar retrouvés,
publications récupérées, plafond de publications par chercheur) ; ils ne constituent pas une
mesure exhaustive de la production scientifique de la FSBM.
"""
from __future__ import annotations

import logging
import textwrap
from pathlib import Path
from typing import Any, Optional

import matplotlib

matplotlib.use("Agg")  # rendu sans écran (scripts / CI)
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from src.data.extract_fsbm_members import is_target_institution  # noqa: E402

logger = logging.getLogger(__name__)
BAR_COLOR = "#2F5D8A"
ACCENT = "#C8553D"
DISCLAIMER = "Dataset collecté — ne représente pas la production totale de la FSBM"


def _style() -> None:
    plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 160, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25,
                         "axes.axisbelow": True, "font.size": 10, "axes.titleweight": "bold"})


def _wrap(label: Any, width: int = 42) -> str:
    return textwrap.shorten(str(label), width=width, placeholder="…")


def _save(fig: plt.Figure, path: Path) -> Path:
    fig.text(0.99, -0.02, DISCLAIMER, ha="right", va="top", fontsize=7, color="#666666")  # sous les axes (bbox tight)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _barh(series: pd.Series, title: str, xlabel: str, path: Path, color: str = BAR_COLOR, top: int = 15) -> Optional[Path]:
    series = series.dropna().sort_values(ascending=False).head(top)
    if series.empty:
        logger.warning("Figure ignorée (aucune donnée) : %s", title)
        return None
    _style()
    fig, ax = plt.subplots(figsize=(9, max(3, 0.42 * len(series) + 1.2)))
    ax.barh([_wrap(i) for i in series.index][::-1], series.values[::-1], color=color)
    for y, v in enumerate(series.values[::-1]):
        ax.text(v, y, f" {int(v):,}".replace(",", " "), va="center", fontsize=8)
    ax.set_title(title, loc="left")
    ax.set_xlabel(xlabel)
    ax.grid(axis="y", visible=False)
    return _save(fig, path)


# --------------------------------------------------------------------------- tables
def explode_researcher_labs(researchers: pd.DataFrame) -> pd.DataFrame:
    """Une ligne par (chercheur, laboratoire) : les affectations multiples sont séparées par « | »."""
    df = researchers.copy()
    df["laboratoire"] = df["laboratoire"].fillna("(non renseigné)").astype(str).str.split(" | ", regex=False)
    return df.explode("laboratoire")


def publications_per_researcher(researchers: pd.DataFrame, links: pd.DataFrame) -> pd.DataFrame:
    counts = links.groupby("chercheur_id")["article_id"].nunique().rename("n_publications")
    df = researchers[["chercheur_id", "nom_complet", "laboratoire"]].merge(counts, on="chercheur_id", how="left")
    df["n_publications"] = df["n_publications"].fillna(0).astype(int)
    return df.sort_values("n_publications", ascending=False)


def publications_per_laboratory(researchers: pd.DataFrame, links: pd.DataFrame) -> pd.Series:
    """Nombre d'articles distincts par laboratoire (un article co-signé par 2 labos compte pour chacun)."""
    merged = links.merge(explode_researcher_labs(researchers)[["chercheur_id", "laboratoire"]], on="chercheur_id")
    return merged.groupby("laboratoire")["article_id"].nunique().sort_values(ascending=False)


def interests_frequency(researchers: pd.DataFrame) -> pd.Series:
    """Fréquence des domaines d'intérêt (insensible à la casse) parmi les profils Scholar retrouvés."""
    def as_list(value: Any) -> list[str]:
        if isinstance(value, list):
            return value
        if isinstance(value, str) and value.strip():
            return [v.strip() for v in value.split(";")]
        return []

    items = [i.lower() for value in researchers["interests"] for i in as_list(value)]
    return pd.Series(items).value_counts() if items else pd.Series(dtype=int)


# --------------------------------------------------------------------------- pipeline d'analyse
def run_descriptive_analysis(members: pd.DataFrame, researchers: pd.DataFrame, articles: pd.DataFrame,
                             links: pd.DataFrame, figures_dir: Path, reports_dir: Path,
                             institution: str = "FSBM") -> dict[str, Any]:
    """Génère toutes les figures + ``descriptive_summary.json`` ; retourne les chemins et les tables clés."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, Any] = {"figures": []}
    add = lambda p: out["figures"].append(str(p)) if p else None  # noqa: E731

    fsbm = members[is_target_institution(members["etablissement"], institution) & ~members["is_duplicate_row"].astype(bool)]
    per_lab = fsbm.drop_duplicates(["chercheur_id", "laboratoire"]).groupby(fsbm["laboratoire"].fillna("(non renseigné)"))["chercheur_id"].nunique()
    add(_barh(per_lab, f"Chercheurs {institution} par laboratoire", "Nombre de chercheurs", figures_dir / "01_chercheurs_par_laboratoire.png"))

    per_res = publications_per_researcher(researchers, links)
    add(_barh(per_res.set_index("nom_complet")["n_publications"], "Publications par chercheur (top 15)",
              "Publications distinctes dans le dataset", figures_dir / "02_publications_par_chercheur.png"))
    per_lab_pubs = publications_per_laboratory(researchers, links)
    add(_barh(per_lab_pubs, "Publications par laboratoire", "Publications distinctes", figures_dir / "03_publications_par_laboratoire.png", ACCENT))

    matched = researchers[researchers["scholar_profile_status"] == "matched"]
    add(_barh(matched.set_index("nom_complet")["citations_totales"], "Citations totales Scholar par chercheur (top 15)",
              "Citations (profil Scholar)", figures_dir / "04_citations_par_chercheur.png"))

    years = articles["annee"].dropna().astype(int).value_counts().sort_index()
    if not years.empty:
        _style()
        fig, ax = plt.subplots(figsize=(9, 4))
        full = years.reindex(range(years.index.min(), years.index.max() + 1), fill_value=0)
        ax.bar(full.index, full.values, color=BAR_COLOR)
        ax.set_title("Publications par année (évolution annuelle)", loc="left")
        ax.set_xlabel("Année")
        ax.set_ylabel("Publications")
        add(_save(fig, figures_dir / "05_publications_par_annee.png"))

    h = matched["h_index"].dropna().astype(int)
    if not h.empty:
        _style()
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(h, bins=min(15, max(3, h.nunique())), color=BAR_COLOR, edgecolor="white")
        ax.set_title(f"Distribution du h-index (profils récupérés, n = {len(h)})", loc="left")
        ax.set_xlabel("h-index (Google Scholar)")
        ax.set_ylabel("Chercheurs")
        add(_save(fig, figures_dir / "06_distribution_h_index.png"))

    add(_barh(interests_frequency(matched), "Domaines d'intérêt les plus fréquents (profils Scholar)",
              "Nombre de chercheurs", figures_dir / "07_domaines_interet.png", ACCENT, top=20))

    coverage = articles["abstract_status"].value_counts()
    if not coverage.empty:
        _style()
        fig, ax = plt.subplots(figsize=(8, 3.2))
        ax.barh(coverage.index[::-1], coverage.values[::-1], color=BAR_COLOR)
        share = 100 * (articles["abstract_clean"].notna().sum()) / len(articles)
        ax.set_title(f"Statut des abstracts — {share:.1f} % des publications ont un abstract", loc="left")
        ax.set_xlabel("Publications")
        ax.grid(axis="y", visible=False)
        add(_save(fig, figures_dir / "08_couverture_abstracts.png"))

    summary = {
        "n_chercheurs_fsbm": int(len(researchers)),
        "n_profils_scholar_retrouves": int(len(matched)),
        "n_publications_uniques": int(len(articles)),
        "taux_publications_avec_abstract_pct": round(100 * articles["abstract_clean"].notna().mean(), 1) if len(articles) else None,
        "top_chercheurs_par_publications": per_res.head(10)[["nom_complet", "n_publications"]].to_dict("records"),
        "publications_par_laboratoire": per_lab_pubs.to_dict(),
        "publications_par_annee": {int(k): int(v) for k, v in years.items()},
        "h_index": {"n": int(len(h)), "median": float(h.median()) if len(h) else None,
                    "max": int(h.max()) if len(h) else None},
        "avertissement": DISCLAIMER,
    }
    pd.Series(per_lab, name="chercheurs").to_csv(reports_dir / "table_chercheurs_par_laboratoire.csv")
    per_res.to_csv(reports_dir / "table_publications_par_chercheur.csv", index=False)
    from src.utils.io import write_json
    write_json(reports_dir / "descriptive_summary.json", summary)
    out["summary"] = summary
    logger.info("%d figures générées dans %s", len(out["figures"]), figures_dir)
    return out
