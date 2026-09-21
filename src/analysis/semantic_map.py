"""Cartographie sémantique : projection 2D des embeddings + clustering + étiquetage des groupes.

Les clusters sont des GROUPES SÉMANTIQUES obtenus mathématiquement (KMeans sur les embeddings
normalisés) ; ils ne sont pas des « vérités scientifiques » ni une taxonomie officielle des domaines.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer  # noqa: E402
from sklearn.metrics import silhouette_score  # noqa: E402

from src.search.vector_store import l2_normalize  # noqa: E402

logger = logging.getLogger(__name__)
RANDOM_STATE = 42
FRENCH_STOP_WORDS = {"le", "la", "les", "un", "une", "des", "du", "de", "et", "en", "au", "aux", "dans", "pour",
                     "par", "sur", "avec", "est", "sont", "que", "qui", "ce", "cette", "ces", "se", "sa", "son",
                     "ses", "ou", "plus", "nous", "cet", "ainsi", "comme", "entre", "leur", "leurs"}
# mots génériques des titres/abstracts scientifiques : écartés de l'ÉTIQUETAGE des clusters uniquement (jamais des embeddings)
ACADEMIC_GENERIC = {"study", "studies", "using", "based", "analysis", "approach", "paper", "results", "method", "methods", "new",
                    "use", "used", "case", "effect", "effects", "review", "towards", "toward", "novel", "et", "al", "de", "des", "en"}
STOP_WORDS = list(ENGLISH_STOP_WORDS | FRENCH_STOP_WORDS | ACADEMIC_GENERIC)  # utilisés UNIQUEMENT pour étiqueter les clusters


# --------------------------------------------------------------------------- réduction & clustering
def reduce_to_2d(embeddings: np.ndarray, method: str = "umap", random_state: int = RANDOM_STATE,
                 n_neighbors: int = 15, min_dist: float = 0.1) -> tuple[np.ndarray, str]:
    """Projette en 2D. Retourne ``(coordonnées, méthode réellement utilisée)`` (repli PCA tracé, jamais silencieux)."""
    x = l2_normalize(embeddings)
    n = len(x)
    if n < 3:
        raise ValueError("Au moins 3 publications sont nécessaires pour une cartographie.")
    if method == "umap":
        try:
            import umap

            reducer = umap.UMAP(n_components=2, n_neighbors=max(2, min(n_neighbors, n - 1)), min_dist=min_dist,
                                metric="cosine", random_state=random_state)
            return reducer.fit_transform(x), "umap"
        except Exception as exc:  # noqa: BLE001
            logger.warning("UMAP indisponible/échec (%s) : repli sur PCA — la méthode utilisée est enregistrée.", exc)
    return PCA(n_components=2, random_state=random_state).fit_transform(x), "pca"


def cluster_embeddings(embeddings: np.ndarray, n_clusters: Optional[int] = None, k_range: tuple[int, int] = (4, 12),
                       random_state: int = RANDOM_STATE) -> tuple[np.ndarray, int, dict[int, float]]:
    """KMeans ; si ``n_clusters`` est None, k est choisi par silhouette (cosinus) sur ``k_range``."""
    x = l2_normalize(embeddings)
    n = len(x)
    scores: dict[int, float] = {}
    if n_clusters is None:
        low, high = max(2, k_range[0]), min(k_range[1], n - 1)
        if high < low:
            n_clusters = max(1, min(low, n))
        else:
            sample = np.random.RandomState(random_state).choice(n, size=min(n, 4000), replace=False)
            for k in range(low, high + 1):
                labels = KMeans(n_clusters=k, n_init=10, random_state=random_state).fit_predict(x)
                scores[k] = float(silhouette_score(x[sample], labels[sample], metric="cosine"))
            n_clusters = max(scores, key=scores.get)
    labels = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state).fit_predict(x)
    return labels, int(n_clusters), scores


def describe_clusters(texts: list[str], titles: list[str], embeddings: np.ndarray, labels: np.ndarray,
                      n_terms: int = 6, n_titles: int = 3) -> pd.DataFrame:
    """Résumé par cluster : taille, termes TF-IDF les plus caractéristiques, titres les plus centraux."""
    x = l2_normalize(embeddings)
    vectorizer = TfidfVectorizer(stop_words=STOP_WORDS, ngram_range=(1, 2), min_df=2 if len(texts) > 30 else 1,
                                 max_features=30000, lowercase=True)
    try:
        tfidf = vectorizer.fit_transform(texts)
        vocab = np.array(vectorizer.get_feature_names_out())
    except ValueError:  # vocabulaire vide
        tfidf, vocab = None, np.array([])
    rows = []
    for cluster in sorted(set(labels)):
        members = np.where(labels == cluster)[0]
        centroid = l2_normalize(x[members].mean(axis=0))[0]
        closest = members[np.argsort(-(x[members] @ centroid))[:n_titles]]
        terms: list[str] = []
        if tfidf is not None and len(vocab):
            mean_scores = np.asarray(tfidf[members].mean(axis=0)).ravel()
            terms = [vocab[i] for i in mean_scores.argsort()[::-1][:n_terms] if mean_scores[i] > 0]
        rows.append({"cluster": int(cluster), "n_publications": int(len(members)),
                     "top_terms": ", ".join(terms), "titres_representatifs": " || ".join(titles[i] for i in closest),
                     "label": f"C{cluster} · " + (", ".join(terms[:3]) if terms else "(sans terme)")})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- figures
def _plotly_map(df: pd.DataFrame, color_col: str, title: str, path: Path) -> Path:
    import plotly.express as px

    fig = px.scatter(df, x="x", y="y", color=color_col, hover_name="titre",
                     hover_data={"x": False, "y": False, "annee": True, "chercheurs": True, "laboratoire": True,
                                 "cluster_label": True},
                     title=title, color_discrete_sequence=px.colors.qualitative.Safe, opacity=0.8)
    fig.update_traces(marker={"size": 7})
    fig.update_layout(legend_title_text="", xaxis_title="axe 1", yaxis_title="axe 2", template="plotly_white",
                      annotations=[{"text": "Groupes sémantiques obtenus mathématiquement (non des catégories scientifiques officielles)",
                                    "xref": "paper", "yref": "paper", "x": 0, "y": -0.12, "showarrow": False,
                                    "font": {"size": 10, "color": "#666"}}])
    fig.write_html(path, include_plotlyjs="cdn")
    return path


def _matplotlib_map(df: pd.DataFrame, title: str, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(11, 8))
    cmap = plt.get_cmap("tab10" if df["cluster"].nunique() <= 10 else "tab20")
    for i, (cluster, sub) in enumerate(df.groupby("cluster")):
        ax.scatter(sub["x"], sub["y"], s=16, alpha=0.75, color=cmap(i % cmap.N), label=sub["cluster_label"].iloc[0])
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_xlabel("axe 1")
    ax.set_ylabel("axe 2")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8, frameon=False)
    fig.text(0.01, 0.005, "Groupes sémantiques obtenus mathématiquement (KMeans sur embeddings zembed-1), "
                          "pas des catégories scientifiques officielles.", fontsize=7, color="#666")
    fig.savefig(path, bbox_inches="tight", dpi=160)
    plt.close(fig)
    return path


def build_semantic_map(articles: pd.DataFrame, embeddings: np.ndarray, links: pd.DataFrame, researchers: pd.DataFrame,
                       figures_dir: Path, reports_dir: Path, method: str = "umap", n_clusters: Optional[int] = None,
                       k_range: tuple[int, int] = (4, 12), n_neighbors: int = 15, min_dist: float = 0.1,
                       random_state: int = RANDOM_STATE) -> dict[str, Any]:
    """Pipeline complet de cartographie ; ``articles`` et ``embeddings`` doivent être alignés ligne à ligne."""
    if len(articles) != len(embeddings):
        raise ValueError(f"Alignement impossible : {len(articles)} articles pour {len(embeddings)} embeddings")
    figures_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    coords, used_method = reduce_to_2d(embeddings, method, random_state, n_neighbors, min_dist)
    labels, k, silhouettes = cluster_embeddings(embeddings, n_clusters, k_range, random_state)
    titles = articles["titre"].tolist()
    texts = articles["embedding_text"].fillna(articles["titre"]).tolist()
    summary = describe_clusters(texts, titles, embeddings, labels)

    names = researchers.set_index("chercheur_id")
    per_article = links.groupby("article_id")["chercheur_id"].apply(list).to_dict()
    def _names(article_id: str) -> str:
        return "; ".join(names.loc[c, "nom_complet"] for c in per_article.get(article_id, []) if c in names.index)
    def _lab(article_id: str) -> str:
        labs = [str(names.loc[c, "laboratoire"]).split(" | ")[0] for c in per_article.get(article_id, [])
                if c in names.index and pd.notna(names.loc[c, "laboratoire"])]
        return labs[0] if labs else "(non renseigné)"

    df = pd.DataFrame({"article_id": articles["article_id"].values, "titre": titles, "annee": articles["annee"].values,
                       "x": coords[:, 0], "y": coords[:, 1], "cluster": labels})
    df["cluster_label"] = df["cluster"].map(dict(zip(summary["cluster"], summary["label"])))
    df["chercheurs"] = df["article_id"].map(_names)
    df["laboratoire"] = df["article_id"].map(_lab)
    df["annee"] = df["annee"].astype("object").where(df["annee"].notna(), None)

    paths = {
        "html_clusters": _plotly_map(df, "cluster_label", "Cartographie sémantique des publications FSBM — groupes",
                                     figures_dir / "semantic_map_clusters.html"),
        "html_laboratoires": _plotly_map(df, "laboratoire", "Cartographie sémantique — couleur = laboratoire",
                                         figures_dir / "semantic_map_laboratoires.html"),
        "png_clusters": _matplotlib_map(df, f"Cartographie sémantique ({used_method.upper()}, {k} groupes)",
                                        figures_dir / "semantic_map_clusters.png"),
    }
    df.to_csv(reports_dir / "semantic_map_points.csv", index=False)
    summary.to_csv(reports_dir / "cluster_summary.csv", index=False)
    info = {"reduction_method_used": used_method, "reduction_method_requested": method, "n_clusters": k,
            "silhouette_by_k": silhouettes, "random_state": random_state, "n_publications": len(df)}
    return {"points": df, "clusters": summary, "paths": {k_: str(v) for k_, v in paths.items()}, "info": info}
