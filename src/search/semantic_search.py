"""Recherche sémantique de publications et de chercheurs FSBM (basée uniquement sur les embeddings).

requête → embedding zembed-1 (input_type="query") → similarité cosinus → top-K publications ;
les chercheurs sont classés en agrégeant les similarités de leurs meilleures publications retrouvées.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from src.embeddings.zembed_client import EmbeddingClient
from src.search.vector_store import VectorStore

ABSTRACT_PREVIEW_CHARS = 300


@dataclass
class SearchResult:
    """Une publication retrouvée."""

    rank: int
    score: float
    article_id: str
    titre: str
    chercheurs: list[str]
    chercheur_ids: list[str]
    annee: Optional[int]
    journal: Optional[str]
    laboratoires: list[str]
    equipes: list[str]
    citations: Optional[int]
    abstract_court: Optional[str]
    doi: Optional[str]
    url: Optional[str]
    embedding_source: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ResearcherResult:
    """Un chercheur classé par agrégation de ses publications les plus proches de la requête."""

    rank: int
    chercheur_id: str
    nom_complet: str
    score: float
    best_score: float
    n_matching_publications: int
    laboratoire: Optional[str]
    equipe: Optional[str]
    top_publications: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def aggregate_researcher_scores(hits: list[tuple[str, float]], researchers_by_article: dict[str, list[str]],
                                top_m: int = 3) -> dict[str, dict[str, Any]]:
    """Score chercheur = moyenne des ``top_m`` meilleures similarités de ses publications retrouvées.

    Les places manquantes comptent pour 0 : un chercheur avec une seule publication très proche est
    donc moins bien classé qu'un chercheur avec plusieurs publications proches (évite les coups de chance).
    """
    per_researcher: dict[str, list[tuple[str, float]]] = {}
    for article_id, score in hits:
        for cid in researchers_by_article.get(article_id, []):
            per_researcher.setdefault(cid, []).append((article_id, score))
    out: dict[str, dict[str, Any]] = {}
    for cid, items in per_researcher.items():
        items.sort(key=lambda t: t[1], reverse=True)
        best = items[:top_m]
        out[cid] = {"score": sum(s for _, s in best) / top_m, "best_score": best[0][1],
                    "n_matching": len(items), "top_articles": best}
    return out


def _short(text: Optional[str], limit: int = ABSTRACT_PREVIEW_CHARS) -> Optional[str]:
    if not text or (isinstance(text, float) and text != text):
        return None
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0] + "…"


def _none(value: Any) -> Any:
    return None if value is None or (not isinstance(value, (list, str)) and pd.isna(value)) else value


class SemanticSearchEngine:
    """Moteur de recherche : index vectoriel + client d'embedding + tables de métadonnées."""

    def __init__(self, store: VectorStore, client: EmbeddingClient, publications: pd.DataFrame,
                 researchers: pd.DataFrame, links: pd.DataFrame) -> None:
        self.store, self.client = store, client
        self.publications = publications.set_index("article_id", drop=False)
        self.researchers = researchers.set_index("chercheur_id", drop=False)
        self.researchers_by_article: dict[str, list[str]] = links.groupby("article_id")["chercheur_id"].apply(list).to_dict()

    @classmethod
    def from_directories(cls, vector_store_dir: Path | str, processed_dir: Path | str,
                         client: EmbeddingClient) -> "SemanticSearchEngine":
        """Charge index + tables nettoyées depuis le disque."""
        processed = Path(processed_dir)
        return cls(VectorStore.load(vector_store_dir), client,
                   pd.read_parquet(processed / "publications_clean.parquet"),
                   pd.read_csv(processed / "chercheurs_clean.csv"),
                   pd.read_csv(processed / "researcher_publications.csv"))

    # ------------------------------------------------------------------ publications
    def _embed_query(self, query: str):
        if not query or not query.strip():
            raise ValueError("La requête est vide.")
        return self.client.embed_queries([query.strip()])

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """Top-K publications les plus proches de la requête (similarité cosinus)."""
        hits = self.store.search(self._embed_query(query), top_k)[0]
        return [self._to_result(rank, article_id, score) for rank, (article_id, score, _) in enumerate(hits, start=1)]

    def _to_result(self, rank: int, article_id: str, score: float) -> SearchResult:
        pub = self.publications.loc[article_id]
        cids = self.researchers_by_article.get(article_id, [])
        people = [self.researchers.loc[c] for c in cids if c in self.researchers.index]
        unique = lambda col: list(dict.fromkeys(str(p[col]) for p in people if _none(p[col]) is not None))  # noqa: E731
        citations = _none(pub["citations"])
        annee = _none(pub["annee"])
        return SearchResult(
            rank=rank, score=round(score, 4), article_id=article_id, titre=pub["titre"],
            chercheurs=[p["nom_complet"] for p in people], chercheur_ids=list(cids),
            annee=int(annee) if annee is not None else None, journal=_none(pub["journal"]) or _none(pub["conference"]),
            laboratoires=unique("laboratoire"), equipes=unique("equipe"),
            citations=int(citations) if citations is not None else None,
            abstract_court=_short(_none(pub["abstract"])), doi=_none(pub["doi"]),
            url=f"https://doi.org/{pub['doi']}" if _none(pub["doi"]) else _none(pub["scholar_url"]),
            embedding_source=_none(pub.get("embedding_source")))

    # ------------------------------------------------------------------ chercheurs
    def search_researchers(self, query: str, top_k: int = 5, pool: int = 100, top_m: int = 3) -> list[ResearcherResult]:
        """Top-K chercheurs : agrégation des similarités de leurs meilleures publications parmi les ``pool`` plus proches."""
        hits = [(a, s) for a, s, _ in self.store.search(self._embed_query(query), pool)[0]]
        scores = aggregate_researcher_scores(hits, self.researchers_by_article, top_m)
        ranked = sorted(scores.items(), key=lambda kv: (kv[1]["score"], kv[1]["best_score"]), reverse=True)[:top_k]
        results = []
        for rank, (cid, agg) in enumerate(ranked, start=1):
            row = self.researchers.loc[cid]
            results.append(ResearcherResult(
                rank=rank, chercheur_id=cid, nom_complet=row["nom_complet"], score=round(agg["score"], 4),
                best_score=round(agg["best_score"], 4), n_matching_publications=agg["n_matching"],
                laboratoire=_none(row["laboratoire"]), equipe=_none(row["equipe"]),
                top_publications=[{"article_id": a, "titre": self.publications.loc[a, "titre"], "score": round(s, 4)}
                                  for a, s in agg["top_articles"]]))
        return results


_DEFAULT_ENGINE: Optional[SemanticSearchEngine] = None


def get_default_engine() -> SemanticSearchEngine:
    """Construit (une fois) le moteur par défaut à partir de ``config/settings.yaml`` et de ``.env``."""
    global _DEFAULT_ENGINE
    if _DEFAULT_ENGINE is None:
        from src.embeddings.zembed_client import ZEmbedClient
        from src.utils.config import load_settings, resolve_path

        settings = load_settings()
        client = ZEmbedClient.from_settings(settings, cache_path=resolve_path(settings, "vector_store_dir") / "embedding_cache.sqlite")
        _DEFAULT_ENGINE = SemanticSearchEngine.from_directories(
            resolve_path(settings, "vector_store_dir"), resolve_path(settings, "processed_dir"), client)
    return _DEFAULT_ENGINE


def semantic_search(query: str, top_k: int = 5, engine: Optional[SemanticSearchEngine] = None) -> list[dict[str, Any]]:
    """``semantic_search(query, top_k=5)`` : requête → zembed-1 → cosinus → top-K publications (dicts)."""
    engine = engine or get_default_engine()
    return [r.to_dict() for r in engine.search(query, top_k)]
