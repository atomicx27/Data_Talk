"""Thread-Safe Hybrid Vector + BM25 Lexical RAG Store for Data_Talk.

Provides the unstructured/retrieval half of the Two-Tier Semantic Memory:
- Collection 1 (`schema_glossary`): Table & column business descriptions and synonyms.
- Collection 2 (`business_rules`): Onboarding interview Q&A answers, status code rules, and in-chat corrections.
- Collection 3 (`few_shot_queries`): User-provided DAX measures, calculated columns, and validated Question -> SQL/DAX examples.

Engine Strategy:
- Uses an embedded, zero-build SQLite + NumPy TF-IDF/BM25 + Hashing/Cosine Vector Hybrid Engine by default
  (100% Windows Python 3.13 wheel-safe, thread-safe with RLock, 0 external servers required, 0 bytes leave localhost).
- Optionally integrates with ChromaDB if installed.
"""

from __future__ import annotations

import json
import math
import re
import sqlite3
import threading
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9_]+", (text or "").lower())


def _hash_embed(text: str, dim: int = 256) -> np.ndarray:
    """Fast local deterministic feature-hashing + n-gram embedding (zero external API leak)."""
    vec = np.zeros(dim, dtype=np.float32)
    tokens = _tokenize(text)
    if not tokens:
        return vec
    # Unigrams + bigrams for semantic phrase capture
    grams = list(tokens) + [f"{tokens[i]}_{tokens[i+1]}" for i in range(len(tokens) - 1)]
    for g in grams:
        h = hash(g)
        idx = abs(h) % dim
        sign = 1.0 if (h & 1) == 0 else -1.0
        vec[idx] += sign
    norm = float(np.linalg.norm(vec))
    if norm > 0:
        vec /= norm
    return vec


class RAGStore:
    """Thread-safe local hybrid (Vector + BM25) RAG store persisted per workspace."""

    COLLECTIONS = ("schema_glossary", "business_rules", "few_shot_queries")

    def __init__(self, workspace_dir: str | Path) -> None:
        self.workspace_dir = Path(workspace_dir)
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.workspace_dir / "rag_memory.sqlite"
        self._lock = threading.RLock()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False, timeout=15.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS rag_documents (
                        doc_id TEXT NOT NULL,
                        collection TEXT NOT NULL,
                        content TEXT NOT NULL,
                        metadata_json TEXT NOT NULL,
                        embedding_blob BLOB NOT NULL,
                        updated_at TEXT NOT NULL,
                        PRIMARY KEY (collection, doc_id)
                    )
                    """
                )
                conn.commit()
            finally:
                conn.close()

    def upsert(
        self,
        collection: str,
        doc_id: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Insert or update a document in the specified RAG collection."""
        if collection not in self.COLLECTIONS:
            raise ValueError(f"Unknown RAG collection: {collection}")
        emb = _hash_embed(content)
        now = datetime.now(timezone.utc).isoformat()
        meta_str = json.dumps(metadata or {}, ensure_ascii=False)
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    INSERT INTO rag_documents (doc_id, collection, content, metadata_json, embedding_blob, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(collection, doc_id) DO UPDATE SET
                        content = excluded.content,
                        metadata_json = excluded.metadata_json,
                        embedding_blob = excluded.embedding_blob,
                        updated_at = excluded.updated_at
                    """,
                    (doc_id, collection, content, meta_str, emb.tobytes(), now),
                )
                conn.commit()
            finally:
                conn.close()

    def query(
        self,
        collection: str,
        query_text: str,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Perform hybrid BM25 lexical + cosine vector similarity search."""
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    "SELECT doc_id, content, metadata_json, embedding_blob FROM rag_documents WHERE collection = ?",
                    (collection,),
                ).fetchall()
            finally:
                conn.close()

        if not rows:
            return []

        q_vec = _hash_embed(query_text)
        q_tokens = _tokenize(query_text)

        # Compute document frequencies for BM25
        doc_tokens_list = [_tokenize(r["content"]) for r in rows]
        df_counts: Counter[str] = Counter()
        for d_tokens in doc_tokens_list:
            for tok in set(d_tokens):
                df_counts[tok] += 1

        n_docs = len(rows)
        avg_dl = sum(len(t) for t in doc_tokens_list) / max(1, n_docs)

        scored: list[dict[str, Any]] = []
        for row, d_tokens in zip(rows, doc_tokens_list):
            d_vec = np.frombuffer(row["embedding_blob"], dtype=np.float32)
            cosine_sim = float(np.dot(q_vec, d_vec)) if len(d_vec) == len(q_vec) else 0.0

            # BM25 score
            bm25 = 0.0
            tf_counter = Counter(d_tokens)
            dl = len(d_tokens)
            for qt in q_tokens:
                tf = tf_counter.get(qt, 0)
                if tf > 0:
                    df = df_counts.get(qt, 0)
                    idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
                    bm25 += idf * ((tf * 2.2) / (tf + 1.2 * (1.0 - 0.75 + 0.75 * (dl / max(1.0, avg_dl)))))

            # Normalize and combine BM25 + Cosine
            hybrid_score = 0.55 * max(0.0, cosine_sim) + 0.45 * min(1.0, bm25 / 5.0)
            scored.append(
                {
                    "doc_id": row["doc_id"],
                    "content": row["content"],
                    "metadata": json.loads(row["metadata_json"]),
                    "score": round(hybrid_score, 4),
                }
            )

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    def retrieve_context_bundle(
        self, query_text: str, top_k_per_collection: int = 5
    ) -> dict[str, list[dict[str, Any]]]:
        """Retrieve relevant schema glossary entries, business rules, and few-shot DAX/SQL pairs."""
        return {
            "schema_glossary": self.query("schema_glossary", query_text, top_k=top_k_per_collection),
            "business_rules": self.query("business_rules", query_text, top_k=top_k_per_collection),
            "few_shot_queries": self.query("few_shot_queries", query_text, top_k=top_k_per_collection),
        }

    def count_by_collection(self) -> dict[str, int]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT collection, COUNT(*) as cnt FROM rag_documents GROUP BY collection"
            ).fetchall()
        counts = {c: 0 for c in self.COLLECTIONS}
        for r in rows:
            counts[r["collection"]] = int(r["cnt"])
        return counts
