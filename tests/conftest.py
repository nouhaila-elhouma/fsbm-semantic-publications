"""Fixtures partagées. Aucun test ne fait de requête réseau (Scholar, API, zembed-1)."""
from __future__ import annotations

import hashlib
import re
from typing import Sequence

import numpy as np
import pandas as pd
import pytest

from src.embeddings.zembed_client import EmbeddingClient


class BagOfWordsClient(EmbeddingClient):
    """Client de TEST déterministe (hachage de mots) : réservé aux tests unitaires.

    Il ne remplace en aucun cas zembed-1 dans le pipeline ; il permet seulement de vérifier
    la mécanique (cosinus, classement, agrégation) sans réseau.
    """

    model = "test-bag-of-words"
    dimension = 128

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dimension, dtype=np.float32)
        for word in re.findall(r"[a-z0-9]+", text.lower()):
            v[int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dimension] += 1.0
        return v

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        return np.vstack([self._vec(t) for t in texts])

    def embed_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self.embed_documents(texts)


@pytest.fixture
def fake_client() -> BagOfWordsClient:
    return BagOfWordsClient()


@pytest.fixture
def mini_corpus() -> dict[str, pd.DataFrame]:
    """Petit corpus fictif de TEST (titres génériques) pour les tests de recherche."""
    publications = pd.DataFrame([
        {"article_id": "art_00001", "titre": "Deep learning for medical imaging segmentation", "annee": 2021, "journal": "J1",
         "conference": None, "citations": 10, "abstract": "convolutional networks segment tumours in mri images",
         "doi": "10.1000/a1", "scholar_url": None, "embedding_source": "title_abstract",
         "embedding_text": "Deep learning for medical imaging segmentation. convolutional networks segment tumours in mri images"},
        {"article_id": "art_00002", "titre": "Natural language processing of arabic tweets", "annee": 2020, "journal": "J2",
         "conference": None, "citations": 5, "abstract": "sentiment analysis with transformers", "doi": None,
         "scholar_url": "https://scholar.example/x", "embedding_source": "title_abstract",
         "embedding_text": "Natural language processing of arabic tweets. sentiment analysis with transformers"},
        {"article_id": "art_00003", "titre": "Solar cells perovskite materials", "annee": 2019, "journal": None,
         "conference": "Conf3", "citations": None, "abstract": None, "doi": None, "scholar_url": None,
         "embedding_source": "title_only", "embedding_text": "Solar cells perovskite materials"},
        {"article_id": "art_00004", "titre": "Language models for medical text processing", "annee": 2022, "journal": "J4",
         "conference": None, "citations": 1, "abstract": "natural language processing for clinical notes", "doi": None,
         "scholar_url": None, "embedding_source": "title_abstract",
         "embedding_text": "Language models for medical text processing. natural language processing for clinical notes"},
    ])
    researchers = pd.DataFrame([
        {"chercheur_id": "fsbm_a_b", "nom_complet": "A B", "laboratoire": "Lab IA", "equipe": "Eq 1"},
        {"chercheur_id": "fsbm_c_d", "nom_complet": "C D", "laboratoire": "Lab IA", "equipe": None},
        {"chercheur_id": "fsbm_e_f", "nom_complet": "E F", "laboratoire": "Lab Energie", "equipe": "Eq 9"},
    ])
    links = pd.DataFrame([
        {"chercheur_id": "fsbm_a_b", "article_id": "art_00001"}, {"chercheur_id": "fsbm_a_b", "article_id": "art_00004"},
        {"chercheur_id": "fsbm_c_d", "article_id": "art_00002"}, {"chercheur_id": "fsbm_c_d", "article_id": "art_00004"},
        {"chercheur_id": "fsbm_e_f", "article_id": "art_00003"},
    ])
    return {"publications": publications, "researchers": researchers, "links": links}
