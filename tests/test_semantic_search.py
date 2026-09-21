"""Tests : cosinus, index vectoriel (FAISS et NumPy), classement de la recherche sémantique, agrégation chercheurs."""
import numpy as np
import pytest

from src.search.semantic_search import SemanticSearchEngine, aggregate_researcher_scores, semantic_search
from src.search.vector_store import VectorStore, cosine_similarity, faiss, l2_normalize

BACKENDS = ["numpy"] + (["faiss"] if faiss is not None else [])


# ---------------------------------------------------------------- cosinus
def test_cosine_similarity_values():
    a = np.array([[1.0, 0.0], [1.0, 1.0]])
    b = np.array([[2.0, 0.0], [0.0, 3.0], [-1.0, 0.0]])
    sims = cosine_similarity(a, b)
    assert sims.shape == (2, 3)
    np.testing.assert_allclose(sims[0], [1.0, 0.0, -1.0], atol=1e-6)
    np.testing.assert_allclose(sims[1, 0], 1 / np.sqrt(2), atol=1e-6)


def test_l2_normalize_and_zero_vector():
    out = l2_normalize(np.array([[3.0, 4.0], [0.0, 0.0]]))
    np.testing.assert_allclose(out[0], [0.6, 0.8], atol=1e-6)
    assert np.isfinite(out).all() and np.allclose(out[1], 0)


# ---------------------------------------------------------------- index
@pytest.mark.parametrize("backend", BACKENDS)
def test_vector_store_ranking_and_scores(backend):
    vectors = np.array([[1, 0, 0], [0.9, 0.1, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
    store = VectorStore(backend).build(vectors, ["a", "b", "c", "d"])
    (hits,) = store.search(np.array([[5.0, 0.0, 0.0]]), k=3)
    assert [h[0] for h in hits][:2] == ["a", "b"]                # « c » et « d » sont ex æquo (cosinus 0)
    assert hits[0][1] == pytest.approx(1.0, abs=1e-5)
    assert hits[0][1] >= hits[1][1] >= hits[2][1]              # ordre décroissant
    assert hits[1][1] == pytest.approx(cosine_similarity(vectors[[1]], np.array([[1.0, 0, 0]]))[0, 0], abs=1e-5)


@pytest.mark.parametrize("backend", BACKENDS)
def test_vector_store_save_load_roundtrip(backend, tmp_path):
    rng = np.random.default_rng(42)
    vectors = rng.normal(size=(20, 8)).astype(np.float32)
    store = VectorStore(backend).build(vectors, [f"art_{i}" for i in range(20)], [{"i": i} for i in range(20)])
    store.save(tmp_path)
    loaded = VectorStore.load(tmp_path)
    q = rng.normal(size=(1, 8))
    assert store.search(q, 5) == loaded.search(q, 5)
    assert (tmp_path / "id_map.json").exists() and (tmp_path / "index_info.json").exists()
    assert loaded.info["normalized"] is True


def test_vector_store_rejects_bad_input():
    store = VectorStore("numpy")
    with pytest.raises(ValueError):
        store.build(np.ones((2, 3)), ["a", "a"])                      # ids dupliqués
    with pytest.raises(ValueError):
        VectorStore("numpy").build(np.ones((2, 3)), ["a"])            # mauvais nombre d'ids
    with pytest.raises(ValueError):
        VectorStore("numpy").build(np.array([[np.nan, 1.0]]), ["a"])  # NaN
    ok = VectorStore("numpy").build(np.ones((2, 3)), ["a", "b"])
    with pytest.raises(ValueError):
        ok.search(np.ones((1, 4)), 1)                                 # mauvaise dimension


def test_search_k_larger_than_index():
    store = VectorStore("numpy").build(np.eye(3), ["a", "b", "c"])
    assert len(store.search(np.array([[1.0, 0, 0]]), k=50)[0]) == 3


# ---------------------------------------------------------------- moteur de recherche
@pytest.fixture
def engine(fake_client, mini_corpus):
    pubs = mini_corpus["publications"]
    vectors = fake_client.embed_documents(pubs["embedding_text"].tolist())
    store = VectorStore("numpy").build(vectors, pubs["article_id"].tolist())
    return SemanticSearchEngine(store, fake_client, pubs, mini_corpus["researchers"], mini_corpus["links"])


def test_publication_search_ranking(engine):
    results = engine.search("deep learning for medical imaging", top_k=3)
    assert results[0].article_id == "art_00001"
    assert [r.rank for r in results] == [1, 2, 3]
    assert results[0].score >= results[1].score >= results[2].score
    assert results[0].chercheurs == ["A B"] and results[0].laboratoires == ["Lab IA"]
    assert results[0].url == "https://doi.org/10.1000/a1"


def test_search_result_fields_and_missing_values(engine):
    top = engine.search("solar cells perovskite", top_k=1)[0]
    assert top.article_id == "art_00003" and top.citations is None and top.abstract_court is None
    assert top.journal == "Conf3" and top.doi is None and top.laboratoires == ["Lab Energie"]
    shared = {r.article_id: r for r in engine.search("language processing", top_k=4)}["art_00004"]
    assert shared.chercheurs == ["A B", "C D"]       # article co-signé : les deux chercheurs FSBM


def test_empty_query_rejected(engine):
    with pytest.raises(ValueError):
        engine.search("   ")


def test_researcher_ranking_uses_publication_similarity(engine):
    res = engine.search_researchers("natural language processing", top_k=3, pool=4, top_m=2)
    assert res[0].chercheur_id == "fsbm_c_d"          # 2 publications NLP proches (art_00002 et art_00004)
    assert res[-1].chercheur_id in {"fsbm_e_f", "fsbm_a_b"}
    assert res[0].score >= res[1].score
    assert res[0].n_matching_publications == 2 and len(res[0].top_publications) == 2


def test_aggregate_researcher_scores_penalises_single_lucky_hit():
    hits = [("p1", 0.90), ("p2", 0.60), ("p3", 0.55), ("p4", 0.50)]
    owners = {"p1": ["solo"], "p2": ["team"], "p3": ["team"], "p4": ["team"]}
    scores = aggregate_researcher_scores(hits, owners, top_m=3)
    assert scores["solo"]["score"] == pytest.approx(0.30) and scores["solo"]["best_score"] == 0.90
    assert scores["team"]["score"] == pytest.approx((0.60 + 0.55 + 0.50) / 3)
    assert scores["team"]["score"] > scores["solo"]["score"]


def test_semantic_search_function_returns_dicts(engine):
    out = semantic_search("perovskite solar", top_k=2, engine=engine)
    assert isinstance(out[0], dict) and out[0]["article_id"] == "art_00003" and {"rank", "score", "titre"} <= set(out[0])
