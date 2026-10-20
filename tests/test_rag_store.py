"""Tests for hybrid RAG store, stable hashing, RRF fusion, and migration."""

import os
import subprocess
import sys
from pathlib import Path
import numpy as np

from src.knowledge.rag_store import EMBED_DIM, RAGStore, _hash_embed


def test_hash_embed_stability_across_processes():
    """Verify that embeddings are invariant to Python hash salt changes."""
    script = (
        "import numpy as np; "
        "from src.knowledge.rag_store import _hash_embed; "
        "vec = _hash_embed('test query orders revenue'); "
        "print(','.join(f'{x:.6f}' for x in vec[:5]))"
    )

    env1 = os.environ.copy()
    env1["PYTHONHASHSEED"] = "1234"
    proc1 = subprocess.run([sys.executable, "-c", script], env=env1, capture_output=True, text=True, check=True)

    env2 = os.environ.copy()
    env2["PYTHONHASHSEED"] = "9999"
    proc2 = subprocess.run([sys.executable, "-c", script], env=env2, capture_output=True, text=True, check=True)

    assert proc1.stdout.strip() == proc2.stdout.strip()


def test_rag_store_upsert_and_rrf_query(tmp_path: Path):
    store = RAGStore(tmp_path)
    store.upsert(
        "business_rules",
        "rule_1",
        "Cancelled orders have ord_stat equal to X and should be excluded from net revenue.",
        {"filter": "ord_stat <> 'X'"},
    )
    store.upsert(
        "business_rules",
        "rule_2",
        "Gross margin is calculated as (rev_net_usd - cogs_usd) / rev_net_usd.",
        {"filter": ""},
    )

    # Lexical match on "cancelled orders"
    hits = store.query("business_rules", "How are cancelled orders defined?", top_k=2)
    assert len(hits) == 2
    assert hits[0]["doc_id"] == "rule_1"
    assert "score" in hits[0]
    assert "bm25" in hits[0]
    assert "cosine" in hits[0]
    assert hits[0]["score"] > hits[1]["score"]


def test_rag_store_cache_invalidation(tmp_path: Path):
    store = RAGStore(tmp_path)
    store.upsert("business_rules", "r1", "Alpha query documentation")
    # Query populates cache
    assert len(store.query("business_rules", "Alpha")) == 1

    # Upsert invalidates cache and new doc appears
    store.upsert("business_rules", "r2", "Beta query documentation")
    assert len(store.query("business_rules", "Beta")) >= 1
    assert store.count_by_collection()["business_rules"] == 2
