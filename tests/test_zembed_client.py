"""Tests : client zembed-1 (modèle imposé, lots, retries/429, cache/reprise, dimension) et enrichisseur d'abstracts.

Aucun appel réseau : le SDK / la session HTTP sont simulés.
"""
from __future__ import annotations

import numpy as np
import pytest
import zeroentropy

from src.data.publication_enricher import (PublicationEnricher, best_title_match, cache_key, clean_api_abstract,
                                           openalex_abstract)
from src.embeddings.zembed_client import (EmbeddingCache, EmbeddingError, MissingAPIKeyError, ZEmbedClient)
from src.utils.config import load_settings


# ---------------------------------------------------------------- faux SDK
class _Result:
    def __init__(self, embedding):
        self.embedding = embedding


class _Response:
    def __init__(self, vectors):
        self.results = [_Result(v) for v in vectors]
        self.usage = type("U", (), {"total_tokens": 7, "total_bytes": 1})()


class FakeModels:
    def __init__(self, dim=8, failures=(), fail_batch_index=None):
        self.dim, self.failures, self.calls, self.fail_batch_index = dim, list(failures), [], fail_batch_index

    def embed(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail_batch_index is not None and len(self.calls) == self.fail_batch_index:
            raise zeroentropy.AuthenticationError("bad key", response=_http_response(401), body=None)
        if self.failures:
            raise self.failures.pop(0)
        return _Response([[float(len(t) % 5 + i) for i in range(self.dim)] for t in kwargs["input"]])


class FakeSDK:
    def __init__(self, **kw):
        self.models = FakeModels(**kw)


def _http_response(status, headers=None):
    import httpx

    return httpx.Response(status, request=httpx.Request("POST", "https://api.zeroentropy.dev/v1/models/embed"), headers=headers or {})


def make_client(tmp_path=None, sdk=None, **kw):
    sleeps = []
    client = ZEmbedClient(api_key="k", sdk_client=sdk or FakeSDK(), sleep=sleeps.append,
                          cache_path=(tmp_path / "cache.sqlite") if tmp_path else None, **kw)
    client.sleeps = sleeps
    return client


# ---------------------------------------------------------------- exigences zembed-1
@pytest.mark.parametrize("bad_model", ["text-embedding-3-small", "zembed-2", "ZEMBED-1", ""])
def test_model_is_never_silently_replaced(bad_model):
    with pytest.raises(ValueError, match="zembed-1"):
        ZEmbedClient(api_key="k", model=bad_model, sdk_client=FakeSDK())


def test_missing_api_key_gives_actionable_error(monkeypatch):
    monkeypatch.delenv("ZEROENTROPY_API_KEY", raising=False)
    with pytest.raises(MissingAPIKeyError, match="ZEROENTROPY_API_KEY"):
        ZEmbedClient()


def test_invalid_options_rejected():
    with pytest.raises(ValueError):
        ZEmbedClient(api_key="k", dimensions=999, sdk_client=FakeSDK())
    with pytest.raises(ValueError):
        ZEmbedClient(api_key="k", latency="turbo", sdk_client=FakeSDK())


def test_documents_and_queries_use_the_right_input_type_and_model():
    sdk = FakeSDK(dim=640)
    client = make_client(sdk=sdk, dimensions=640)
    client.embed_documents(["a doc"])
    client.embed_queries(["a query"])
    assert [c["input_type"] for c in sdk.models.calls] == ["document", "query"]
    assert all(c["model"] == "zembed-1" and c["dimensions"] == 640 for c in sdk.models.calls)


def test_dimension_is_read_from_response_not_assumed():
    client = make_client(sdk=FakeSDK(dim=11))
    assert client.dimension is None
    assert client.embed_documents(["x", "y"]).shape == (2, 11) and client.dimension == 11


def test_requested_dimension_mismatch_is_detected():
    client = make_client(sdk=FakeSDK(dim=8), dimensions=640)
    with pytest.raises(EmbeddingError, match="640"):
        client.embed_documents(["x"])


def test_batching_by_count_and_bytes():
    client = make_client(batch_size=3)
    batches = client.make_batches([f"t{i}" for i in range(7)])
    assert [len(b) for b in batches] == [3, 3, 1]
    big = ["x" * 1_500_000] * 3                        # 3 × ~1,5 Mo > 4 Mo → 2 requêtes malgré batch_size=32
    assert [len(b) for b in make_client(batch_size=32).make_batches(big)] == [2, 1]


def test_output_order_matches_input_and_duplicates_are_embedded_once():
    sdk = FakeSDK()
    client = make_client(sdk=sdk, batch_size=2)
    out = client.embed_documents(["aa", "bbbbb", "aa", "cc"])
    assert out.shape[0] == 4 and np.array_equal(out[0], out[2])
    assert sum(len(c["input"]) for c in sdk.models.calls) == 3


def test_empty_texts_rejected():
    with pytest.raises(ValueError):
        make_client().embed_documents(["ok", "  "])


# ---------------------------------------------------------------- retry / rate limits
def test_rate_limit_429_is_retried_with_backoff_and_retry_after():
    limited = zeroentropy.RateLimitError("slow down", response=_http_response(429, {"retry-after": "42"}), body=None)
    sdk = FakeSDK(failures=[limited, zeroentropy.APITimeoutError(request=None)])
    client = make_client(sdk=sdk, backoff_base=1)
    client.embed_documents(["x"])
    assert client.stats["retries"] == 2 and len(client.sleeps) == 2 and client.sleeps[0] >= 42


def test_gives_up_after_max_retries_and_reports_resume_hint():
    sdk = FakeSDK(failures=[zeroentropy.APITimeoutError(request=None)] * 10)
    client = make_client(sdk=sdk, retries=2, backoff_base=0)
    with pytest.raises(EmbeddingError, match="relancez"):
        client.embed_documents(["x"])
    assert len(sdk.models.calls) == 3


def test_auth_error_is_fatal_and_not_retried():
    sdk = FakeSDK(fail_batch_index=1)
    with pytest.raises(EmbeddingError, match="non récupérable"):
        make_client(sdk=sdk).embed_documents(["x"])
    assert len(sdk.models.calls) == 1


# ---------------------------------------------------------------- cache et reprise
def test_progressive_cache_allows_resume_after_failure(tmp_path):
    texts = [f"document number {i}" for i in range(6)]
    sdk = FakeSDK(fail_batch_index=3)                   # le 3e lot échoue (erreur fatale)
    client = make_client(tmp_path, sdk=sdk, batch_size=2)
    with pytest.raises(EmbeddingError):
        client.embed_documents(texts)
    assert len(client.cache) == 4                       # 2 premiers lots sauvegardés

    resumed_sdk = FakeSDK()
    resumed = make_client(tmp_path, sdk=resumed_sdk, batch_size=2)
    out = resumed.embed_documents(texts)
    assert out.shape == (6, 8) and sum(len(c["input"]) for c in resumed_sdk.models.calls) == 2   # seuls les 2 manquants
    assert resumed.stats["cache_hits"] == 4


def test_cache_key_depends_on_model_dimensions_input_type_and_text():
    k = EmbeddingCache.make_key
    assert len({k("zembed-1", None, "query", "t"), k("zembed-1", None, "document", "t"),
                k("zembed-1", 640, "document", "t"), k("zembed-1", None, "document", "t2")}) == 4


# ---------------------------------------------------------------- transport HTTP documenté
class FakeHttp:
    def __init__(self, status=200):
        self.status, self.posts = status, []

    def post(self, url, json, timeout, headers):
        self.posts.append((url, json, headers))
        dim = json.get("dimensions", 3)
        body = {"results": [{"embedding": [0.1] * dim} for _ in json["input"]], "usage": {"total_bytes": 1, "total_tokens": 2}}
        return type("R", (), {"status_code": self.status, "headers": {}, "text": "err", "json": lambda self_: body})()


def test_http_transport_follows_documented_api():
    http = FakeHttp()
    client = ZEmbedClient(api_key="secret", transport="http", http_session=http, dimensions=640, latency="slow")
    assert client.embed_queries(["q"]).shape == (1, 640)
    url, body, headers = http.posts[0]
    assert url == "https://api.zeroentropy.dev/v1/models/embed" and headers == {"Authorization": "Bearer secret"}
    assert body == {"model": "zembed-1", "input": ["q"], "input_type": "query", "dimensions": 640, "latency": "slow"}


# ---------------------------------------------------------------- transport local (poids officiels)
class FakeLocalModel:
    """Imite sentence-transformers : encode_query / encode_document (pas de téléchargement)."""

    def __init__(self, dim=6, bad=False):
        self.dim, self.bad, self.calls = dim, bad, []

    def _encode(self, kind, texts, **kw):
        self.calls.append((kind, list(texts), kw))
        out = np.ones((len(texts), self.dim), dtype=np.float32) * (1 if kind == "doc" else 2)
        if self.bad:
            out[0, 0] = np.nan
        return out

    def encode_query(self, texts, **kw):
        return self._encode("query", texts, **kw)

    def encode_document(self, texts, **kw):
        return self._encode("doc", texts, **kw)


def test_local_transport_needs_no_api_key_and_uses_query_vs_document_encoders(monkeypatch):
    monkeypatch.delenv("ZEROENTROPY_API_KEY", raising=False)
    model = FakeLocalModel()
    client = ZEmbedClient(transport="local", local_model=model)               # aucune clé requise
    docs, queries = client.embed_documents(["a doc", "another"]), client.embed_queries(["a query"])
    assert docs.shape == (2, 6) and queries.shape == (1, 6) and client.dimension == 6
    assert [c[0] for c in model.calls] == ["doc", "query"] and float(docs[0, 0]) == 1 and float(queries[0, 0]) == 2


def test_local_transport_keeps_the_model_guard_and_a_fixed_weights_id():
    from src.embeddings.zembed_client import LOCAL_MODEL_ID

    assert LOCAL_MODEL_ID == "zeroentropy/zembed-1-embedding"
    with pytest.raises(ValueError, match="zembed-1"):
        ZEmbedClient(transport="local", model="all-MiniLM-L6-v2", local_model=FakeLocalModel())
    with pytest.raises(ValueError):
        ZEmbedClient(transport="local", dtype="int4", local_model=FakeLocalModel())


def test_local_transport_rejects_non_finite_embeddings_with_actionable_hint():
    client = ZEmbedClient(transport="local", local_model=FakeLocalModel(bad=True))
    with pytest.raises(EmbeddingError, match="dtype"):
        client.embed_documents(["x"])


def test_local_transport_shares_the_cache_and_resumes(tmp_path):
    first = ZEmbedClient(transport="local", local_model=FakeLocalModel(), cache_path=tmp_path / "c.sqlite", batch_size=2)
    first.embed_documents(["t1", "t2", "t3"])
    model = FakeLocalModel()
    second = ZEmbedClient(transport="local", local_model=model, cache_path=tmp_path / "c.sqlite", batch_size=2)
    second.embed_documents(["t1", "t2", "t3", "t4"])
    assert sum(len(c[1]) for c in model.calls) == 1 and second.stats["cache_hits"] == 3     # seul « t4 » est recalculé


def test_default_settings_use_the_local_open_weights():
    settings = load_settings()
    assert settings["embedding"]["model"] == "zembed-1" and settings["embedding"]["transport"] == "local"


# ---------------------------------------------------------------- enrichisseur
def test_openalex_inverted_index_reconstruction():
    assert openalex_abstract({"Deep": [0], "learning": [1, 4], "is": [2], "great": [3]}) == "Deep learning is great learning"
    assert openalex_abstract(None) is None


def test_clean_api_abstract_strips_jats_and_label():
    assert clean_api_abstract("<jats:p>Abstract  We study <jats:italic>CO2</jats:italic> capture.</jats:p>") == "We study CO2 capture."


def test_best_title_match_requires_similar_title_and_year():
    pub = {"titre": "Machine learning algorithms for breast cancer prediction and diagnosis", "annee": 2021}
    items = [{"t": "Deep learning for cats", "y": 2021}, {"t": "Machine learning algorithms for breast cancer prediction and diagnosis", "y": 2021},
             {"t": "Machine learning algorithms for breast cancer prediction and diagnosis", "y": 2005}]
    hit = best_title_match(pub, items, lambda i: i["t"], lambda i: i["y"], 0.9)
    assert hit is items[1]
    assert best_title_match(pub, items[:1], lambda i: i["t"], lambda i: i["y"], 0.9) is None


class FakeApiSession:
    """Répond selon l'URL ; enregistre les appels."""

    def __init__(self, routes):
        self.routes, self.urls, self.headers = routes, [], {}

    def get(self, url, params=None, headers=None, timeout=None):
        self.urls.append(url)
        for prefix, (status, payload) in self.routes.items():
            if url.startswith(prefix):
                return type("R", (), {"status_code": status, "json": lambda s, p=payload: p, "raise_for_status": lambda s: None})()
        return type("R", (), {"status_code": 404, "json": lambda s: {}, "raise_for_status": lambda s: None})()


def make_enricher(routes, tmp_path, **cfg_over):
    settings = load_settings()
    settings["enrichment"].update({"retries": 0, "sources": ["crossref", "openalex"], **cfg_over})
    return PublicationEnricher(settings, tmp_path / "enrich.json", session=FakeApiSession(routes), delay_fn=lambda: None)


def raw_pub(**kw):
    return {"raw_pub_id": "c::1", "titre": "Machine learning algorithms for breast cancer prediction and diagnosis", "annee": 2021,
            "external_url": None, "pdf_url": None, "doi": None, "abstract": None, "abstract_status": "not_found", **kw}


LONG_ABSTRACT = "This is a complete abstract about breast cancer prediction with five machine learning algorithms. " * 2


def test_enrichment_finds_doi_and_abstract_via_crossref_title_search(tmp_path):
    payload = {"message": {"items": [{"DOI": "10.1016/J.PROCS.2021.07.001", "abstract": f"<jats:p>{LONG_ABSTRACT}</jats:p>",
                                      "title": ["Machine learning algorithms for breast cancer prediction and diagnosis"],
                                      "issued": {"date-parts": [[2021, 1, 1]]}}]}}
    enricher = make_enricher({"https://api.crossref.org/works": (200, payload)}, tmp_path)
    pub = raw_pub()
    assert enricher.enrich_publication(pub) is True
    assert pub["doi"] == "10.1016/j.procs.2021.07.001" and pub["abstract_source"] == "crossref" and pub["abstract_status"] == "found"
    assert pub["abstract"] == LONG_ABSTRACT.strip()


def test_enrichment_never_fabricates_when_nothing_is_found(tmp_path):
    enricher = make_enricher({}, tmp_path)                      # toutes les API répondent 404
    pub = raw_pub()
    enricher.enrich_publication(pub)
    assert pub["abstract"] is None and pub["doi"] is None and pub["abstract_status"] == "not_found"
    assert [a["status"] for a in pub["abstract_attempts"]] == ["not_found", "not_found"]


def test_enrichment_flags_api_errors_and_stays_retryable(tmp_path):
    enricher = make_enricher({"https://api.crossref.org": (503, {}), "https://api.openalex.org": (503, {})}, tmp_path)
    pub = raw_pub()
    assert enricher.enrich_publication(pub) is False            # non définitif → sera retenté à la reprise
    assert pub["abstract_status"] == "api_error" and pub["abstract"] is None


def test_scholar_full_abstract_is_kept_and_only_doi_is_looked_up(tmp_path):
    payload = {"message": {"items": [{"DOI": "10.1000/zzz", "abstract": "<p>" + "other abstract text " * 10 + "</p>",
                                      "title": [raw_pub()["titre"]], "issued": {"date-parts": [[2021]]}}]}}
    enricher = make_enricher({"https://api.crossref.org/works": (200, payload)}, tmp_path)
    pub = raw_pub(abstract=LONG_ABSTRACT, abstract_status="found", abstract_source="google_scholar")
    enricher.enrich_publication(pub)
    assert pub["abstract"] == LONG_ABSTRACT and pub["abstract_source"] == "google_scholar" and pub["doi"] == "10.1000/zzz"


def test_truncated_scholar_abstract_is_replaced_by_longer_full_one(tmp_path):
    payload = {"message": {"items": [{"DOI": "10.1000/zzz", "abstract": LONG_ABSTRACT, "title": [raw_pub()["titre"]],
                                      "issued": {"date-parts": [[2021]]}}]}}
    enricher = make_enricher({"https://api.crossref.org/works": (200, payload)}, tmp_path)
    pub = raw_pub(abstract="This is a complete abstract …", abstract_status="truncated", abstract_source="google_scholar")
    enricher.enrich_publication(pub)
    assert pub["abstract_status"] == "found" and pub["abstract_source"] == "crossref"


def test_enrich_all_uses_cache_between_researchers_and_runs(tmp_path):
    payload = {"message": {"items": [{"DOI": "10.1000/zzz", "abstract": LONG_ABSTRACT, "title": [raw_pub()["titre"]],
                                      "issued": {"date-parts": [[2021]]}}]}}
    enricher = make_enricher({"https://api.crossref.org/works": (200, payload)}, tmp_path)
    a, b = raw_pub(raw_pub_id="c1::1"), raw_pub(raw_pub_id="c2::9")           # même article, 2 chercheurs
    stats = enricher.enrich_all([a, b])
    assert stats["enriched"] == 1 and stats["from_cache"] == 1 and b["doi"] == "10.1000/zzz"
    again = make_enricher({}, tmp_path)                                        # nouveau process : cache disque
    assert again.enrich_all([raw_pub()])["from_cache"] == 1
    assert cache_key(a) == cache_key(b)


# ---------------------------------------------------------------- qualité : affiliations ≠ abstract, revue/éditeur
def test_affiliation_lists_are_not_accepted_as_abstracts():
    from src.preprocessing.text_cleaner import looks_like_affiliation

    aff = ("1Research and Engineering Laboratory, National High School for Electricity and Mechanics, Hassan II University "
           "of Casablanca, Casablanca, Morocco 2Laboratory of Information Processing, Faculty of Sciences Ben M’Sick, "
           "Hassan II University of Casablanca, Casablanca, Morocco")
    assert looks_like_affiliation(aff)
    assert clean_api_abstract(f"<jats:p>{aff}</jats:p>") is None
    real = ("We study air pollution in Casablanca, Morocco, and propose a method based on machine learning; "
            "results show that particulate matter increases near the university campus.")
    assert not looks_like_affiliation(real) and clean_api_abstract(real) == real
    assert not looks_like_affiliation(None) and not looks_like_affiliation("Deep learning for medical imaging.")


def test_affiliation_abstract_is_treated_as_missing_by_the_cleaner():
    from src.preprocessing.data_cleaner import clean_publication_record

    rec = clean_publication_record({"raw_pub_id": "c::1", "chercheur_id": "fsbm_a_b", "titre": "A title", "auteurs": ["X"],
                                    "annee": 2022, "abstract": "1Laboratory of Physics, Faculty of Sciences, Hassan II University, Morocco",
                                    "abstract_status": "found", "abstract_source": "crossref"})
    assert rec["abstract"] is None and rec["abstract_clean"] is None and rec["abstract_status"] == "not_found"
    assert rec["embedding_source"] == "title_only"


def test_enrichment_fills_missing_journal_and_publisher_but_never_overwrites(tmp_path):
    item = {"DOI": "10.1000/zzz", "abstract": LONG_ABSTRACT, "title": [raw_pub()["titre"]], "issued": {"date-parts": [[2021]]},
            "container-title": ["Procedia Computer Science"], "publisher": "Elsevier BV"}
    enricher = make_enricher({"https://api.crossref.org/works": (200, {"message": {"items": [item]}})}, tmp_path)
    pub = raw_pub()
    enricher.enrich_publication(pub)
    assert pub["journal"] == "Procedia Computer Science" and pub["publisher"] == "Elsevier BV"
    kept = raw_pub(journal="Journal déjà connu par Scholar", publisher="Éditeur Scholar")
    make_enricher({"https://api.crossref.org/works": (200, {"message": {"items": [item]}})}, tmp_path).enrich_publication(kept)
    assert kept["journal"] == "Journal déjà connu par Scholar" and kept["publisher"] == "Éditeur Scholar"


def test_old_cache_entries_without_journal_are_re_enriched(tmp_path):
    item = {"DOI": "10.1000/zzz", "abstract": LONG_ABSTRACT, "title": [raw_pub()["titre"]], "issued": {"date-parts": [[2021]]},
            "container-title": ["Some Journal"]}
    enricher = make_enricher({"https://api.crossref.org/works": (200, {"message": {"items": [item]}})}, tmp_path)
    old = {"doi": "10.1000/zzz", "abstract": LONG_ABSTRACT, "abstract_source": "crossref", "abstract_status": "found",
           "abstract_attempts": [], "final": True}                                             # entrée v1 : pas de « v », pas de revue
    enricher.cache[cache_key(raw_pub())] = old
    pub = raw_pub()
    assert enricher.enrich_all([pub])["from_cache"] == 0 and pub["journal"] == "Some Journal"
