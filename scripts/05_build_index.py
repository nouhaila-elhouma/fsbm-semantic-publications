"""Étape 3b — Construction de l'index vectoriel FAISS (cosinus) depuis les embeddings zembed-1.

Entrées : data/vector_store/embeddings.npy + embedding_manifest.csv, data/processed/publications_clean.parquet
Sorties : data/vector_store/{index.faiss, id_map.json, metadata.json, index_info.json}

Exemple :
    python scripts/05_build_index.py
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from _bootstrap import ROOT  # noqa: F401

from src.search.vector_store import VectorStore
from src.utils.config import load_settings, resolve_path
from src.utils.io import read_json
from src.utils.logger import setup_logging


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backend", choices=["auto", "faiss", "numpy"], default="auto", help="Moteur d'index (défaut : faiss si installé)")
    p.add_argument("--config", type=Path, help="Fichier settings.yaml alternatif")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    settings = load_settings(args.config)
    log = setup_logging("05_build_index", resolve_path(settings, "logs_dir"))
    store_dir = resolve_path(settings, "vector_store_dir")
    npy, manifest_csv = store_dir / "embeddings.npy", store_dir / "embedding_manifest.csv"
    parquet = resolve_path(settings, "processed_dir") / "publications_clean.parquet"
    if not (npy.exists() and manifest_csv.exists() and parquet.exists()):
        log.error("Fichiers manquants : lancez scripts/03 puis scripts/04 avant l'index.")
        return 1

    vectors = np.load(npy)
    manifest = pd.read_csv(manifest_csv)
    articles = pd.read_parquet(parquet).set_index("article_id", drop=False)
    if len(manifest) != len(vectors):
        log.error("Manifest (%d lignes) et embeddings (%d) désalignés : relancez scripts/04.", len(manifest), len(vectors))
        return 1

    cols = ["article_id", "titre", "annee", "journal", "doi"]
    meta = []
    for article_id, source in zip(manifest["article_id"], manifest["embedding_source"]):
        row = articles.loc[article_id, cols]
        meta.append({**{c: (None if pd.isna(row[c]) else (int(row[c]) if c == "annee" else row[c])) for c in cols},
                     "embedding_source": source})
    run_info = read_json(store_dir / "embedding_run.json", default={})
    store = VectorStore(args.backend).build(vectors, manifest["article_id"].tolist(), meta,
                                            info={"model": run_info.get("model", settings["embedding"]["model"]),
                                                  "source_manifest": manifest_csv.name})
    store.save(store_dir)
    log.info("Index construit et sauvegardé dans %s", store_dir)
    print(f"\n✔ Index {store.backend} : {len(store)} vecteurs, dimension {store.dimension}, métrique cosinus (vecteurs normalisés L2)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
