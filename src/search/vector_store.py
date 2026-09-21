"""Index vectoriel local : FAISS (``IndexFlatIP``) sur vecteurs L2-normalisés = similarité cosinus exacte.

Si ``faiss`` n'est pas installé, un repli NumPy exact (mêmes résultats, produit matriciel) est utilisé ;
le backend réellement employé est enregistré dans ``index_info.json``.
Fichiers sauvegardés : ``index.faiss`` (ou ``vectors.npy``), ``id_map.json`` (vector_id → article_id),
``metadata.json`` et ``index_info.json``.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

try:  # dépendance optionnelle à l'exécution
    import faiss  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - dépend de l'environnement
    faiss = None

logger = logging.getLogger(__name__)


def l2_normalize(vectors: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Normalise chaque ligne à la norme 1 (float32). Un vecteur nul reste nul."""
    array = np.atleast_2d(np.asarray(vectors, dtype=np.float32))
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    return array / np.maximum(norms, eps)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Matrice de similarités cosinus entre les lignes de ``a`` (n, d) et de ``b`` (m, d) → (n, m)."""
    return l2_normalize(a) @ l2_normalize(b).T


class VectorStore:
    """Index de similarité cosinus avec correspondance ``vector_id → article_id`` et métadonnées."""

    def __init__(self, backend: str = "auto") -> None:
        if backend not in {"auto", "faiss", "numpy"}:
            raise ValueError("backend doit valoir 'auto', 'faiss' ou 'numpy'")
        if backend == "faiss" and faiss is None:
            raise ImportError("faiss n'est pas installé (pip install faiss-cpu) ; utilisez backend='numpy'.")
        self.backend = "faiss" if (backend in {"auto", "faiss"} and faiss is not None) else "numpy"
        self.ids: list[str] = []
        self.metadata: list[dict[str, Any]] = []
        self.dimension: Optional[int] = None
        self.info: dict[str, Any] = {}
        self._index: Any = None
        self._matrix: Optional[np.ndarray] = None

    def __len__(self) -> int:
        return len(self.ids)

    # ------------------------------------------------------------------ construction
    def build(self, vectors: np.ndarray, ids: Sequence[str], metadata: Optional[Sequence[dict[str, Any]]] = None,
              info: Optional[dict[str, Any]] = None) -> "VectorStore":
        """Normalise les vecteurs et construit l'index."""
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or len(vectors) != len(ids):
            raise ValueError(f"Incohérence : {vectors.shape} vecteurs pour {len(ids)} identifiants")
        if len(set(ids)) != len(ids):
            raise ValueError("Les article_id doivent être uniques")
        if not np.isfinite(vectors).all():
            raise ValueError("Les embeddings contiennent des NaN/inf")
        normalized = l2_normalize(vectors)
        self.dimension = int(normalized.shape[1])
        self.ids = list(ids)
        self.metadata = list(metadata) if metadata is not None else [{} for _ in ids]
        if self.backend == "faiss":
            self._index = faiss.IndexFlatIP(self.dimension)
            self._index.add(normalized)
        else:
            self._matrix = normalized
        self.info = {"backend": self.backend, "dimension": self.dimension, "n_vectors": len(self.ids),
                     "metric": "cosine (inner product on L2-normalized vectors)", "normalized": True,
                     "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), **(info or {})}
        return self

    # ------------------------------------------------------------------ recherche
    def search(self, query_vectors: np.ndarray, k: int = 5) -> list[list[tuple[str, float, dict[str, Any]]]]:
        """Top-k par requête : liste de ``(article_id, score_cosinus, métadonnées)`` triée par score décroissant."""
        if not self.ids:
            return [[] for _ in np.atleast_2d(query_vectors)]
        queries = l2_normalize(query_vectors)
        if queries.shape[1] != self.dimension:
            raise ValueError(f"Dimension de requête {queries.shape[1]} ≠ dimension de l'index {self.dimension}")
        k = max(1, min(k, len(self.ids)))
        if self.backend == "faiss":
            scores, indices = self._index.search(queries, k)
        else:
            sims = queries @ self._matrix.T
            indices = np.argsort(-sims, axis=1)[:, :k]
            scores = np.take_along_axis(sims, indices, axis=1)
        return [[(self.ids[i], float(s), self.metadata[i]) for s, i in zip(row_s, row_i) if i >= 0]
                for row_s, row_i in zip(scores, indices)]

    # ------------------------------------------------------------------ persistance
    def save(self, directory: Path | str) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        if self.backend == "faiss":
            faiss.write_index(self._index, str(directory / "index.faiss"))
        else:
            np.save(directory / "vectors.npy", self._matrix)
        (directory / "id_map.json").write_text(
            json.dumps({str(i): a for i, a in enumerate(self.ids)}, ensure_ascii=False, indent=1), encoding="utf-8")
        (directory / "metadata.json").write_text(json.dumps(self.metadata, ensure_ascii=False, default=str), encoding="utf-8")
        (directory / "index_info.json").write_text(json.dumps(self.info, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path | str) -> "VectorStore":
        directory = Path(directory)
        info = json.loads((directory / "index_info.json").read_text(encoding="utf-8"))
        store = cls(backend=info["backend"])
        id_map = json.loads((directory / "id_map.json").read_text(encoding="utf-8"))
        store.ids = [id_map[str(i)] for i in range(len(id_map))]
        store.metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        store.dimension, store.info = int(info["dimension"]), info
        if info["backend"] == "faiss":
            store._index = faiss.read_index(str(directory / "index.faiss"))
        else:
            store._matrix = np.load(directory / "vectors.npy")
        return store
