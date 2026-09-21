"""Étape 3a — Embeddings zembed-1 des publications (titre + abstract nettoyé).

zembed-1 s'exécute en LOCAL avec les poids officiels ``zeroentropy/zembed-1-embedding`` (Hugging Face, Apache-2.0,
révision épinglée) : aucune clé API n'est nécessaire (ZeroEntropy n'accepte plus de nouvelles inscriptions).
Un GPU est recommandé (modèle de 4 Md de paramètres) : sur un PC sans GPU, lancez ce script sur Google Colab
(voir notebooks/02_colab_embeddings_gpu.ipynb) puis récupérez les fichiers de data/vector_store/.

Sorties (data/vector_store/) : embeddings.npy (n × d, float32), embedding_manifest.csv, embedding_run.json,
embedding_cache.sqlite (cache/reprise, contient aussi les embeddings des requêtes de démonstration).

Exemples :
    python scripts/04_generate_embeddings.py --check            # charge le modèle et affiche la dimension
    python scripts/04_generate_embeddings.py --limit 20         # essai sur 20 publications
    python scripts/04_generate_embeddings.py --with-queries     # tout + les 5 requêtes de démonstration
    python scripts/04_generate_embeddings.py --attach-only      # intègre embeddings.npy dans dataset_final.json
"""
import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from _bootstrap import ROOT  # noqa: F401

from src.embeddings.zembed_client import EmbeddingError, ZEmbedClient
from src.search.demo_queries import DEMO_QUERIES
from src.utils.config import load_settings, resolve_path
from src.utils.io import read_json, write_json
from src.utils.logger import setup_logging


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--check", action="store_true", help="Charge zembed-1 et affiche la dimension des vecteurs, puis s'arrête")
    p.add_argument("--limit", type=int, metavar="N", help="N'encoder que les N premières publications (mode test)")
    p.add_argument("--with-queries", action="store_true", help="Encode aussi les 5 requêtes de démonstration (stockées dans le cache)")
    p.add_argument("--attach-only", action="store_true", help="Ne rien encoder : intégrer embeddings.npy existant dans dataset_final.json")
    p.add_argument("--transport", choices=["local", "sdk", "http"], help="Défaut : settings.yaml (local = poids officiels)")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], help="Transport local : périphérique")
    p.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], help="Transport local : précision")
    p.add_argument("--batch-size", type=int, help="Taille des lots (défaut : settings.yaml)")
    p.add_argument("--max-seq-length", type=int, help="Transport local : nombre maximal de tokens par texte")
    p.add_argument("--dimensions", type=int, help="Dimension de sortie (2560, 1280, 640, 320, 160, 80, 40) ; défaut : celle du modèle")
    p.add_argument("--embed-in-json", choices=["true", "false", "auto"], help="Inclure les vecteurs dans dataset_final.json")
    p.add_argument("--config", type=Path, help="Fichier settings.yaml alternatif")
    return p.parse_args()


def attach_embeddings_to_dataset(dataset: list[dict], vectors_by_article: dict[str, np.ndarray], decimals: int = 6) -> int:
    """Remplit ``embedding_zembed1`` de chaque article ; retourne le nombre d'articles renseignés."""
    n = 0
    for researcher in dataset:
        for article in researcher["articles"]:
            vec = vectors_by_article.get(article["article_id"])
            article["embedding_zembed1"] = [round(float(x), decimals) for x in vec] if vec is not None else None
            n += vec is not None
    return n


def maybe_attach_to_json(settings: dict, article_ids: list[str], vectors: np.ndarray, mode_override: str | None, log) -> None:
    """dataset_final.json : vecteurs inclus ou non (fichier très volumineux en 2560 dimensions)."""
    dataset_path = resolve_path(settings, "processed_dir") / "dataset_final.json"
    dataset = read_json(dataset_path)
    if not dataset:
        return
    mode = mode_override or str(settings["embedding"]["embed_in_json"]).lower()
    n_entries = sum(len(r["articles"]) for r in dataset)
    include = mode == "true" or (mode == "auto" and n_entries * vectors.shape[1] <= settings["embedding"]["embed_in_json_max_floats"])
    if include:
        n = attach_embeddings_to_dataset(dataset, dict(zip(article_ids, vectors)))
        write_json(dataset_path, dataset, indent=None)
        log.info("Embeddings intégrés à dataset_final.json (%d articles).", n)
    else:
        log.info("Embeddings NON intégrés à dataset_final.json (%d entrées × %d dim > limite) : ils sont dans "
                 "embeddings.npy. Forcer avec --embed-in-json true.", n_entries, vectors.shape[1])


def main() -> int:
    args = parse_args()
    settings = load_settings(args.config)
    emb = settings["embedding"]
    for key, value in (("transport", args.transport), ("device", args.device), ("dtype", args.dtype),
                       ("batch_size", args.batch_size), ("max_seq_length", args.max_seq_length), ("dimensions", args.dimensions)):
        if value:
            emb[key] = value
    log = setup_logging("04_generate_embeddings", resolve_path(settings, "logs_dir"))
    store_dir = resolve_path(settings, "vector_store_dir")
    store_dir.mkdir(parents=True, exist_ok=True)

    if args.attach_only:
        npy, manifest_csv = store_dir / "embeddings.npy", store_dir / "embedding_manifest.csv"
        if not (npy.exists() and manifest_csv.exists()):
            log.error("embeddings.npy / embedding_manifest.csv introuvables dans %s", store_dir)
            return 1
        maybe_attach_to_json(settings, pd.read_csv(manifest_csv)["article_id"].tolist(), np.load(npy), args.embed_in_json, log)
        return 0

    try:
        client = ZEmbedClient.from_settings(settings, cache_path=store_dir / "embedding_cache.sqlite")
    except Exception as exc:  # clé absente (sdk/http), modèle invalide…
        log.error("%s", exc)
        return 1
    if args.check:
        try:
            dim = client.verify_connection()
        except EmbeddingError as exc:
            log.error("%s", exc)
            return 1
        print(f"✔ zembed-1 opérationnel (transport {client.transport}) — dimension des vecteurs : {dim} — {client.runtime}")
        return 0

    parquet = resolve_path(settings, "processed_dir") / "publications_clean.parquet"
    if not parquet.exists():
        log.error("%s introuvable : lancez d'abord scripts/03_clean_data.py", parquet)
        return 1
    articles = pd.read_parquet(parquet)
    with_text = articles[articles["embedding_text"].notna() & (articles["embedding_source"] != "none")].reset_index(drop=True)
    if args.limit:
        with_text = with_text.head(args.limit)
    log.info("%d publications à encoder (%d ignorées faute de texte ou hors --limit)", len(with_text), len(articles) - len(with_text))
    if with_text.empty:
        log.error("Aucun texte à encoder.")
        return 1

    def progress(i: int, total: int) -> None:
        log.info("Lot %d/%d encodé (sauvegardé dans le cache)", i, total)

    try:
        vectors = client.embed(with_text["embedding_text"].tolist(), "document", on_batch=progress)
        if args.with_queries:
            client.embed(DEMO_QUERIES, "query")
            log.info("%d requêtes de démonstration encodées et mises en cache", len(DEMO_QUERIES))
    except EmbeddingError as exc:
        log.error("%s", exc)
        return 1

    np.save(store_dir / "embeddings.npy", vectors)
    manifest = pd.DataFrame({
        "row": range(len(with_text)), "article_id": with_text["article_id"], "embedding_source": with_text["embedding_source"],
        "text_sha256": with_text["embedding_text"].map(lambda t: hashlib.sha256(t.encode("utf-8")).hexdigest()),
        "model": client.model, "dimension": vectors.shape[1]})
    manifest.to_csv(store_dir / "embedding_manifest.csv", index=False)
    write_json(store_dir / "embedding_run.json", {
        "model": client.model, "transport": client.transport, "runtime": client.runtime,
        "dimension": int(vectors.shape[1]), "requested_dimensions": client.requested_dimensions,
        "n_embedded": int(len(vectors)), "demo_queries_cached": bool(args.with_queries),
        "embedding_source_counts": with_text["embedding_source"].value_counts().to_dict(), "client_stats": client.stats})
    maybe_attach_to_json(settings, with_text["article_id"].tolist(), vectors, args.embed_in_json, log)
    print(f"\n✔ {len(vectors)} embeddings zembed-1 (dimension {vectors.shape[1]}, transport {client.transport}) — {client.stats}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
