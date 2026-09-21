"""Cartographie sémantique : UMAP/PCA 2D + KMeans + graphiques (matplotlib PNG, Plotly HTML).

Entrées : data/vector_store/embeddings.npy + embedding_manifest.csv, data/processed/*.
Sorties : outputs/figures/semantic_map_*.{png,html}, outputs/reports/{cluster_summary,semantic_map_points}.csv

Exemple :
    python scripts/07_semantic_map.py --method umap
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from _bootstrap import ROOT  # noqa: F401

from src.analysis.semantic_map import build_semantic_map
from src.utils.config import load_settings, resolve_path
from src.utils.io import write_json
from src.utils.logger import setup_logging


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--method", choices=["umap", "pca"], help="Réduction de dimension (défaut : settings.yaml)")
    p.add_argument("--n-clusters", type=int, help="Nombre de clusters (défaut : choisi par silhouette)")
    p.add_argument("--config", type=Path, help="Fichier settings.yaml alternatif")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    settings = load_settings(args.config)
    log = setup_logging("07_semantic_map", resolve_path(settings, "logs_dir"))
    cfg = settings["semantic_map"]
    store_dir, processed = resolve_path(settings, "vector_store_dir"), resolve_path(settings, "processed_dir")
    if not (store_dir / "embeddings.npy").exists():
        log.error("embeddings.npy introuvable : lancez scripts/04_generate_embeddings.py")
        return 1
    vectors = np.load(store_dir / "embeddings.npy")
    manifest = pd.read_csv(store_dir / "embedding_manifest.csv")
    articles = pd.read_parquet(processed / "publications_clean.parquet").set_index("article_id", drop=False).loc[manifest["article_id"]].reset_index(drop=True)
    result = build_semantic_map(
        articles, vectors, pd.read_csv(processed / "researcher_publications.csv"), pd.read_csv(processed / "chercheurs_clean.csv"),
        resolve_path(settings, "figures_dir"), resolve_path(settings, "reports_dir"),
        method=args.method or cfg["reduction"], n_clusters=args.n_clusters or cfg["n_clusters"],
        k_range=tuple(cfg["k_range"]), n_neighbors=cfg["umap_n_neighbors"], min_dist=cfg["umap_min_dist"],
        random_state=settings["project"]["random_state"])
    write_json(resolve_path(settings, "reports_dir") / "semantic_map_info.json", result["info"])
    print("\n=== Cartographie sémantique ===")
    print(f"Réduction : {result['info']['reduction_method_used']} | groupes : {result['info']['n_clusters']} "
          f"| silhouette par k : {result['info']['silhouette_by_k']}")
    print(result["clusters"][["cluster", "n_publications", "top_terms"]].to_string(index=False))
    print("\nFichiers :", *result["paths"].values(), sep="\n  ")
    print("\nNote : les groupes sont des regroupements sémantiques mathématiques, pas des catégories scientifiques officielles.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
