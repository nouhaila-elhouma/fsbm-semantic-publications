"""Client d'embeddings pour le modèle **zembed-1** (fournisseur : ZeroEntropy).

Vérifié dans la documentation officielle (docs.zeroentropy.dev, API « models/embed ») et dans le
SDK installé (paquet ``zeroentropy``) :
  * endpoint : ``POST https://api.zeroentropy.dev/v1/models/embed`` ; auth ``Authorization: Bearer <clé>`` ;
  * corps : ``model`` (« zembed-1 »), ``input`` (chaîne ou liste), ``input_type`` (« query » | « document »,
    OBLIGATOIRE), ``dimensions`` (optionnel : 2560, 1280, 640, 320, 160, 80, 40), ``encoding_format``,
    ``latency`` (« fast » | « slow » | null) ;
  * réponse : ``{"results": [{"embedding": [...]}, ...], "usage": {...}}`` dans l'ordre de l'entrée ;
  * limites : 5 000 000 octets par requête ; coût = ``sum(150 + len(utf8(texte)))`` ; débit par défaut
    500 000 octets/minute (HTTP 429 au-delà → backoff).
La dimension réelle n'est JAMAIS supposée : elle est lue dans la première réponse.

Trois transports, MÊME modèle zembed-1 :
  * ``sdk``   : API hébergée via le SDK officiel ``zeroentropy`` (clé requise — ZeroEntropy n'accepte plus de
                nouvelles inscriptions depuis son rachat) ;
  * ``http``  : la même API hébergée via l'API REST documentée ;
  * ``local`` : POIDS OUVERTS officiels ``zeroentropy/zembed-1-embedding`` (Hugging Face, licence Apache-2.0,
                4 Md de paramètres, base Qwen3-4B) exécutés avec ``sentence-transformers`` — sans clé, sur CPU
                ou GPU (Colab). Fiche officielle : ``model.encode_query`` / ``model.encode_document``,
                dimension par défaut 2560, dimensions Matryoshka 1280…40, contexte 32 768 tokens.

Le modèle n'est jamais remplacé silencieusement : toute autre valeur que « zembed-1 » lève une erreur, et
l'identifiant des poids locaux est une constante (``LOCAL_MODEL_ID``), pas un réglage.
"""
from __future__ import annotations

import hashlib
import logging
import os
import sqlite3
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import numpy as np

from src.utils.retry import backoff_delay

logger = logging.getLogger(__name__)

MODEL_NAME = "zembed-1"
LOCAL_MODEL_ID = "zeroentropy/zembed-1-embedding"   # poids officiels (Hugging Face) — volontairement non configurable
# Révision épinglée (reproductibilité + sécurité : le dépôt fournit un modeling_zembed.py de 20 lignes exécuté via
# trust_remote_code ; relu le 2026-09-21 : il ajoute « <|im_end|>\n » au texte puis tokenise, rien d'autre).
LOCAL_MODEL_REVISION = "cf13c81f3274394053d166740294f7eea4586f7a"
API_URL = "https://api.zeroentropy.dev/v1/models/embed"
PER_TEXT_OVERHEAD_BYTES = 150          # règle de comptage officielle
MAX_REQUEST_BYTES = 4_000_000          # marge sous la limite officielle de 5 000 000
VALID_DIMENSIONS = (2560, 1280, 640, 320, 160, 80, 40)


class EmbeddingError(RuntimeError):
    """Erreur d'embedding non récupérable (le cache conserve tout ce qui a déjà été calculé)."""


class MissingAPIKeyError(EmbeddingError):
    """Clé API absente : définir ZEROENTROPY_API_KEY (fichier .env)."""


class _RetryableApiError(Exception):
    def __init__(self, message: str, retry_after: Optional[float] = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class _FatalApiError(Exception):
    pass


# =========================================================================== interface & cache
class EmbeddingClient(ABC):
    """Abstraction utilisée par la recherche : tout client doit fournir ces deux méthodes."""

    model: str = ""
    dimension: Optional[int] = None

    @abstractmethod
    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        """Vecteurs float32 de forme ``(n, d)`` pour des documents à indexer."""

    @abstractmethod
    def embed_queries(self, texts: Sequence[str]) -> np.ndarray:
        """Vecteurs float32 de forme ``(n, d)`` pour des requêtes utilisateur."""


class EmbeddingCache:
    """Cache disque SQLite (clé = hash du modèle, des dimensions, du type d'entrée et du texte)."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(self.path)
        self._db.execute("CREATE TABLE IF NOT EXISTS embeddings "
                         "(key TEXT PRIMARY KEY, dim INTEGER NOT NULL, vec BLOB NOT NULL)")
        self._db.commit()

    @staticmethod
    def make_key(model: str, dimensions: Optional[int], input_type: str, text: str) -> str:
        raw = f"{model}|{dimensions or 'default'}|{input_type}|{text}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get_many(self, keys: Sequence[str]) -> dict[str, np.ndarray]:
        found: dict[str, np.ndarray] = {}
        for start in range(0, len(keys), 500):
            chunk = list(keys[start:start + 500])
            marks = ",".join("?" * len(chunk))
            for key, dim, blob in self._db.execute(f"SELECT key, dim, vec FROM embeddings WHERE key IN ({marks})", chunk):
                found[key] = np.frombuffer(blob, dtype=np.float32, count=dim).copy()
        return found

    def put_many(self, items: dict[str, np.ndarray]) -> None:
        self._db.executemany("INSERT OR REPLACE INTO embeddings (key, dim, vec) VALUES (?, ?, ?)",
                             [(k, int(v.shape[0]), v.astype(np.float32).tobytes()) for k, v in items.items()])
        self._db.commit()

    def __len__(self) -> int:
        return self._db.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]

    def close(self) -> None:
        self._db.close()


# =========================================================================== client zembed-1
class ZEmbedClient(EmbeddingClient):
    """Client zembed-1 : lots, retries avec backoff, limites de débit, cache et reprise."""

    def __init__(self, api_key: Optional[str] = None, model: str = MODEL_NAME, batch_size: int = 32,
                 dimensions: Optional[int] = None, timeout: float = 60, retries: int = 5, backoff_base: float = 5,
                 latency: Optional[str] = None, transport: str = "sdk", cache_path: Optional[Path | str] = None,
                 sleep: Callable[[float], None] = time.sleep, sdk_client: Any = None, http_session: Any = None,
                 device: str = "auto", dtype: str = "auto", max_seq_length: Optional[int] = 512,
                 local_model: Any = None) -> None:
        if model != MODEL_NAME:
            raise ValueError(f"Modèle exigé : {MODEL_NAME!r} (reçu {model!r}). Aucun remplacement silencieux n'est autorisé.")
        if dimensions is not None and dimensions not in VALID_DIMENSIONS:
            raise ValueError(f"dimensions doit être l'une de {VALID_DIMENSIONS} (ou None), reçu {dimensions}")
        if latency not in (None, "fast", "slow"):
            raise ValueError("latency doit valoir None, 'fast' ou 'slow'")
        if transport not in {"sdk", "http", "local"}:
            raise ValueError("transport doit valoir 'sdk', 'http' ou 'local'")
        if dtype not in {"auto", "bfloat16", "float16", "float32"}:
            raise ValueError("dtype doit valoir 'auto', 'bfloat16', 'float16' ou 'float32'")
        self.device, self.dtype, self.max_seq_length = device, dtype, max_seq_length
        self._local = local_model
        self.runtime: dict[str, Any] = {}                 # périphérique/dtype réellement utilisés (transport local)
        self.api_key = api_key or os.getenv("ZEROENTROPY_API_KEY")
        needs_key = transport in {"sdk", "http"} and sdk_client is None and http_session is None
        if needs_key and not self.api_key:
            raise MissingAPIKeyError("ZEROENTROPY_API_KEY est absente. Créez une clé sur https://dashboard.zeroentropy.dev/ "
                                     "puis renseignez-la dans le fichier .env (voir .env.example).")
        self.model = model
        self.batch_size = max(1, batch_size)
        self.requested_dimensions = dimensions
        self.timeout, self.retries, self.backoff_base = timeout, retries, backoff_base
        self.latency, self.transport = latency, transport
        self.dimension: Optional[int] = None
        self._sleep = sleep
        self.cache = EmbeddingCache(cache_path) if cache_path else None
        self._sdk = sdk_client
        self._http = http_session
        self.stats = {"api_calls": 0, "texts_embedded": 0, "cache_hits": 0, "retries": 0, "total_tokens": 0}

    @classmethod
    def from_settings(cls, settings: dict[str, Any], cache_path: Optional[Path | str] = None) -> "ZEmbedClient":
        """Construit le client depuis ``settings['embedding']`` (la clé vient de l'environnement)."""
        cfg = settings["embedding"]
        return cls(model=cfg["model"], batch_size=cfg["batch_size"], dimensions=cfg.get("dimensions"),
                   timeout=cfg["timeout"], retries=cfg["retries"], backoff_base=cfg["backoff_base"],
                   latency=cfg.get("latency"), transport=cfg.get("transport", "local"), cache_path=cache_path,
                   device=cfg.get("device", "auto"), dtype=cfg.get("dtype", "auto"),
                   max_seq_length=cfg.get("max_seq_length", 512))

    # ------------------------------------------------------------------ transports
    def _load_local(self) -> Any:
        """Charge (une fois) les poids officiels zembed-1 avec sentence-transformers."""
        if self._local is not None:
            return self._local
        try:
            import torch
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise EmbeddingError("transport 'local' : installez torch et sentence-transformers "
                                 "(pip install -r requirements.txt).") from exc
        device = self.device if self.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = self.dtype
        if dtype == "auto":
            # bfloat16 natif à partir d'Ampere (capacité 8.0) ; T4 & co : float16 ; CPU : bfloat16 (float32 = 16 Go de RAM)
            dtype = "float16" if device == "cuda" and torch.cuda.get_device_capability()[0] < 8 else "bfloat16"
        logger.info("Chargement des poids officiels %s (device=%s, dtype=%s, max_seq_length=%s) — le premier "
                    "chargement télécharge ~8 Go", LOCAL_MODEL_ID, device, dtype, self.max_seq_length)
        model = SentenceTransformer(LOCAL_MODEL_ID, trust_remote_code=True, device=device, revision=LOCAL_MODEL_REVISION,
                                    model_kwargs={"torch_dtype": dtype}, truncate_dim=self.requested_dimensions)
        if self.max_seq_length:
            model.max_seq_length = self.max_seq_length
        self.runtime = {"weights": LOCAL_MODEL_ID, "revision": LOCAL_MODEL_REVISION, "device": device, "dtype": dtype, "max_seq_length": self.max_seq_length}
        self._local = model
        return model

    def _call_local(self, texts: list[str], input_type: str) -> tuple[list[list[float]], int]:
        model = self._load_local()
        encode = model.encode_query if input_type == "query" else model.encode_document
        try:
            vectors = encode(texts, batch_size=len(texts), convert_to_numpy=True, show_progress_bar=False)
        except MemoryError as exc:
            raise _FatalApiError("mémoire insuffisante : réduisez embedding.batch_size ou max_seq_length") from exc
        except RuntimeError as exc:                       # ex. torch.cuda.OutOfMemoryError
            if "out of memory" in str(exc).lower():
                raise _FatalApiError("mémoire GPU/CPU insuffisante : réduisez embedding.batch_size ou max_seq_length") from exc
            raise
        array = np.asarray(vectors, dtype=np.float32)
        if not np.isfinite(array).all():
            raise _FatalApiError("embeddings non finis (NaN/inf) : essayez embedding.dtype: bfloat16 (ou float32 si la mémoire le permet)")
        return array.tolist(), 0

    def _call_sdk(self, texts: list[str], input_type: str) -> tuple[list[list[float]], int]:
        import zeroentropy

        if self._sdk is None:
            # max_retries=0 : la politique de retry est gérée ici (une seule couche, pas de double backoff)
            self._sdk = zeroentropy.ZeroEntropy(api_key=self.api_key, timeout=self.timeout, max_retries=0)
        kwargs: dict[str, Any] = {"model": self.model, "input": texts, "input_type": input_type}
        if self.requested_dimensions:
            kwargs["dimensions"] = self.requested_dimensions
        if self.latency:
            kwargs["latency"] = self.latency
        try:
            response = self._sdk.models.embed(**kwargs)
        except zeroentropy.RateLimitError as exc:
            raise _RetryableApiError("HTTP 429 (limite de débit)", _retry_after(exc)) from exc
        except (zeroentropy.APITimeoutError, zeroentropy.APIConnectionError, zeroentropy.InternalServerError) as exc:
            raise _RetryableApiError(f"{type(exc).__name__}: {exc}") from exc
        except zeroentropy.APIStatusError as exc:
            if exc.status_code >= 500:
                raise _RetryableApiError(f"HTTP {exc.status_code}") from exc
            raise _FatalApiError(f"HTTP {exc.status_code} : {exc}") from exc
        usage = getattr(response, "usage", None)
        return [r.embedding for r in response.results], int(getattr(usage, "total_tokens", 0) or 0)

    def _call_http(self, texts: list[str], input_type: str) -> tuple[list[list[float]], int]:
        import requests

        session = self._http or requests
        body: dict[str, Any] = {"model": self.model, "input": texts, "input_type": input_type}
        if self.requested_dimensions:
            body["dimensions"] = self.requested_dimensions
        if self.latency:
            body["latency"] = self.latency
        try:
            resp = session.post(API_URL, json=body, timeout=self.timeout,
                                headers={"Authorization": f"Bearer {self.api_key}"})
        except requests.RequestException as exc:
            raise _RetryableApiError(f"erreur réseau : {exc}") from exc
        if resp.status_code == 429:
            raise _RetryableApiError("HTTP 429 (limite de débit)", _parse_retry_after(resp.headers.get("Retry-After")))
        if resp.status_code >= 500:
            raise _RetryableApiError(f"HTTP {resp.status_code}")
        if resp.status_code >= 400:
            raise _FatalApiError(f"HTTP {resp.status_code} : {resp.text[:300]}")
        data = resp.json()
        return [r["embedding"] for r in data["results"]], int(data.get("usage", {}).get("total_tokens", 0))

    def _embed_batch(self, texts: list[str], input_type: str) -> np.ndarray:
        """Un appel API avec retries/backoff ; valide le nombre et la dimension des vecteurs."""
        call = {"sdk": self._call_sdk, "http": self._call_http, "local": self._call_local}[self.transport]
        for attempt in range(self.retries + 1):
            try:
                self.stats["api_calls"] += 1
                vectors, tokens = call(texts, input_type)
                break
            except _RetryableApiError as exc:
                if attempt >= self.retries:
                    raise EmbeddingError(f"Échec après {attempt + 1} tentatives : {exc}. "
                                         "Les embeddings déjà calculés sont en cache : relancez pour reprendre.") from exc
                delay = max(exc.retry_after or 0, backoff_delay(attempt, self.backoff_base))
                self.stats["retries"] += 1
                logger.warning("zembed-1 : %s — nouvelle tentative %d/%d dans %.0fs", exc, attempt + 1, self.retries, delay)
                self._sleep(delay)
            except _FatalApiError as exc:
                raise EmbeddingError(f"Erreur API zembed-1 non récupérable : {exc}") from exc
        if len(vectors) != len(texts):
            raise EmbeddingError(f"Réponse incohérente : {len(vectors)} vecteurs pour {len(texts)} textes")
        array = np.asarray(vectors, dtype=np.float32)
        if array.ndim != 2:
            raise EmbeddingError(f"Format de réponse inattendu (forme {array.shape})")
        if self.dimension is None:
            self.dimension = int(array.shape[1])
            logger.info("Dimension des embeddings zembed-1 (lue dans la réponse) : %d", self.dimension)
        elif array.shape[1] != self.dimension:
            raise EmbeddingError(f"Dimension inattendue : {array.shape[1]} au lieu de {self.dimension}")
        if self.requested_dimensions and self.dimension != self.requested_dimensions:
            raise EmbeddingError(f"dimensions={self.requested_dimensions} demandé mais {self.dimension} reçu")
        self.stats["total_tokens"] += tokens
        return array

    # ------------------------------------------------------------------ lots
    def make_batches(self, texts: Sequence[str]) -> list[list[str]]:
        """Découpe par nombre (``batch_size``) ET par taille en octets (règle officielle 150 + len(utf8))."""
        batches: list[list[str]] = []
        current: list[str] = []
        current_bytes = 0
        for text in texts:
            cost = PER_TEXT_OVERHEAD_BYTES + len(text.encode("utf-8"))
            if current and (len(current) >= self.batch_size or current_bytes + cost > MAX_REQUEST_BYTES):
                batches.append(current)
                current, current_bytes = [], 0
            current.append(text)
            current_bytes += cost
        if current:
            batches.append(current)
        return batches

    def embed(self, texts: Sequence[str], input_type: str,
              on_batch: Optional[Callable[[int, int], None]] = None) -> np.ndarray:
        """Embeddings dans l'ordre d'entrée, avec cache disque et sauvegarde après chaque lot."""
        if input_type not in {"query", "document"}:
            raise ValueError("input_type doit valoir 'query' ou 'document'")
        texts = list(texts)
        if any(not isinstance(t, str) or not t.strip() for t in texts):
            raise ValueError("Les textes vides ne peuvent pas être encodés (filtrez-les avant l'appel).")
        keys = [EmbeddingCache.make_key(self.model, self.requested_dimensions, input_type, t) for t in texts]
        cached = self.cache.get_many(list(dict.fromkeys(keys))) if self.cache is not None else {}
        self.stats["cache_hits"] += sum(1 for k in keys if k in cached)

        missing = list(dict.fromkeys(t for t, k in zip(texts, keys) if k not in cached))
        batches = self.make_batches(missing)
        for i, batch in enumerate(batches, start=1):
            vectors = self._embed_batch(batch, input_type)
            fresh = {EmbeddingCache.make_key(self.model, self.requested_dimensions, input_type, t): v
                     for t, v in zip(batch, vectors)}
            cached.update(fresh)
            if self.cache is not None:
                self.cache.put_many(fresh)     # sauvegarde progressive : rien n'est perdu si le lot suivant échoue
            self.stats["texts_embedded"] += len(batch)
            if on_batch:
                on_batch(i, len(batches))
        if not texts:
            return np.zeros((0, self.dimension or 0), dtype=np.float32)
        self.dimension = self.dimension or int(next(iter(cached.values())).shape[0])
        return np.vstack([cached[k] for k in keys]).astype(np.float32)

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        return self.embed(texts, "document")

    def embed_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self.embed(texts, "query")

    def verify_connection(self) -> int:
        """Test rapide (1 texte) : valide clé/poids, modèle et dimension. Retourne la dimension."""
        vector = self._embed_batch(["connectivity check"], "query")
        return int(vector.shape[1])


def _parse_retry_after(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _retry_after(exc: Any) -> Optional[float]:
    headers = getattr(getattr(exc, "response", None), "headers", None)
    return _parse_retry_after(headers.get("retry-after")) if headers else None
