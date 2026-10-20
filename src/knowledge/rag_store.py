"""Thread-Safe Hybrid Vector + BM25 Lexical RAG Store for Data_Talk.

Provides the unstructured/retrieval half of the Two-Tier Semantic Memory:
- Collection 1 (`schema_glossary`): Table & column business descriptions and synonyms.
- Collection 2 (`business_rules`): Onboarding interview Q&A answers, status code rules, and in-chat corrections.
- Collection 3 (`few_shot_queries`): User-provided DAX measures, calculated columns, and validated Question -> SQL/DAX examples.

Engine Strategy:
- Uses an embedded, zero-build SQLite + NumPy BM25 + Hashing/Cosine Vector Hybrid Engine by default
  (100% Windows Python 3.13 wheel-safe, thread-safe with RLock, 0 external servers required, 0 bytes leave localhost).
- Lexical (BM25) and vector (cosine) rankings are merged with Reciprocal Rank Fusion (RRF, k=60).
- Embeddings use a *stable* BLAKE2b feature hash so vectors persisted in SQLite stay valid across
  process restarts (Python's built-in ``hash()`` is salted per process and must never be persisted).
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
import threading
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

EMBED_DIM = 256
#: Bump whenever ``_hash_embed`` output changes; stored vectors are rebuilt automatically on open.
EMBED_VERSION = "2"
RRF_K = 60


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9_]+", (text or "").lower())


def _stable_hash(token: str) -> int:
    """Process-independent 64-bit hash (unlike built-in ``hash()``, which is randomised per process)."""
    return int.from_bytes(hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest(), "little")


def _hash_embed(text: str, dim: int = EMBED_DIM) -> np.ndarray:
    """Fast local deterministic feature-hashing + n-gram embedding (zero external API leak)."""
    vec = np.zeros(dim, dtype=np.float32)
    tokens = _tokenize(text)
    if not tokens:
        return vec
    # Unigrams + bigrams for semantic phrase capture
    grams = list(tokens) + [f"{tokens[i]}_{tokens[i + 1]}" for i in range(len(tokens) - 1)]
    for g in grams:
        h = _stable_hash(g)
        idx = h % dim
        sign = 1.0 if (h >> 63) == 0 else -1.0
        vec[idx] += sign
    norm = float(np.linalg.norm(vec))
    if norm > 0:
        vec /= norm
    return vec


@dataclass
class _CollectionIndex:
    """In-memory, per-collection search index (invalidated on every write)."""

    doc_ids: list[str]
    contents: list[str]
    metadata: list[dict[str, Any]]
    matrix: np.ndarray  # (n_docs, dim) float32
    tf: list[Counter[str]]
    doc_len: np.ndarray
    df: Counter[str]
    avg_dl: float


class RAGStore:
    """Thread-safe local hybrid (Vector + BM25, fused with RRF) RAG store persisted per workspace."""

    COLLECTIONS = ("schema_glossary", "business_rules", "few_shot_queries")

    def __init__(self, workspace_dir: str | Path) -> None:
        self.workspace_dir = Path(workspace_dir)
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.workspace_dir / "rag_memory.sqlite"
        self._lock = threading.RLock()
        self._cache: dict[str, _CollectionIndex] = {}
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
                conn.execute("CREATE TABLE IF NOT EXISTS rag_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
                conn.commit()
                self._migrate_embeddings(conn)
            finally:
                conn.close()

    def _migrate_embeddings(self, conn: sqlite3.Connection) -> None:
        """Re-embed every stored document if it was written by an older/unstable embedder."""
        row = conn.execute("SELECT value FROM rag_meta WHERE key = 'embed_version'").fetchone()
        if row is not None and row["value"] == EMBED_VERSION:
            return
        docs = conn.execute("SELECT collection, doc_id, content FROM rag_documents").fetchall()
        with conn:
            for d in docs:
                conn.execute(
                    "UPDATE rag_documents SET embedding_blob = ? WHERE collection = ? AND doc_id = ?",
                    (_hash_embed(d["content"]).tobytes(), d["collection"], d["doc_id"]),
                )
            conn.execute(
                "INSERT INTO rag_meta (key, value) VALUES ('embed_version', ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (EMBED_VERSION,),
            )

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
            self._cache.pop(collection, None)

    def _get_index(self, collection: str) -> _CollectionIndex | None:
        with self._lock:
            cached = self._cache.get(collection)
            if cached is not None:
                return cached
            conn = self._connect()
            try:
                rows = conn.execute(
                    "SELECT doc_id, content, metadata_json, embedding_blob FROM rag_documents "
                    "WHERE collection = ? ORDER BY doc_id",
                    (collection,),
                ).fetchall()
            finally:
                conn.close()
            if not rows:
                return None

            vectors = []
            for r in rows:
                v = np.frombuffer(r["embedding_blob"], dtype=np.float32)
                vectors.append(v if len(v) == EMBED_DIM else _hash_embed(r["content"]))
            tokens = [_tokenize(r["content"]) for r in rows]
            df: Counter[str] = Counter()
            for toks in tokens:
                df.update(set(toks))
            doc_len = np.array([len(t) for t in tokens], dtype=np.float32)
            idx = _CollectionIndex(
                doc_ids=[r["doc_id"] for r in rows],
                contents=[r["content"] for r in rows],
                metadata=[json.loads(r["metadata_json"]) for r in rows],
                matrix=np.vstack(vectors).astype(np.float32),
                tf=[Counter(t) for t in tokens],
                doc_len=doc_len,
                df=df,
                avg_dl=float(doc_len.mean()) if len(doc_len) else 0.0,
            )
            self._cache[collection] = idx
            return idx

    @staticmethod
    def _bm25_scores(idx: _CollectionIndex, q_tokens: list[str], k1: float = 1.2, b: float = 0.75) -> np.ndarray:
        n_docs = len(idx.doc_ids)
        scores = np.zeros(n_docs, dtype=np.float64)
        avg_dl = max(1.0, idx.avg_dl)
        for qt in set(q_tokens):
            df = idx.df.get(qt, 0)
            if df == 0:
                continue
            idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
            for i, tf_counter in enumerate(idx.tf):
                tf = tf_counter.get(qt, 0)
                if tf:
                    denom = tf + k1 * (1.0 - b + b * (idx.doc_len[i] / avg_dl))
                    scores[i] += idf * (tf * (k1 + 1.0)) / denom
        return scores

    @staticmethod
    def _ranks(scores: np.ndarray) -> dict[int, int]:
        """1-based ranks of documents with a strictly positive score (stable ordering)."""
        order = sorted((i for i in range(len(scores)) if scores[i] > 0), key=lambda i: (-scores[i], i))
        return {doc_i: rank for rank, doc_i in enumerate(order, start=1)}

    def query(
        self,
        collection: str,
        query_text: str,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Perform hybrid BM25 lexical + cosine vector search fused with Reciprocal Rank Fusion.

        Each returned hit carries ``score`` (RRF normalised to 0..1), plus the raw ``bm25`` and
        ``cosine`` component scores for transparency.
        """
        idx = self._get_index(collection)
        if idx is None:
            return []

        q_vec = _hash_embed(query_text)
        cosine = idx.matrix @ q_vec
        bm25 = self._bm25_scores(idx, _tokenize(query_text))

        bm25_rank = self._ranks(bm25)
        cos_rank = self._ranks(cosine)
        max_rrf = 2.0 / (RRF_K + 1)

        scored: list[dict[str, Any]] = []
        for i, doc_id in enumerate(idx.doc_ids):
            rrf = 0.0
            if i in bm25_rank:
                rrf += 1.0 / (RRF_K + bm25_rank[i])
            if i in cos_rank:
                rrf += 1.0 / (RRF_K + cos_rank[i])
            scored.append(
                {
                    "doc_id": doc_id,
                    "content": idx.contents[i],
                    "metadata": dict(idx.metadata[i]),
                    "score": round(rrf / max_rrf, 4),
                    "bm25": round(float(bm25[i]), 4),
                    "cosine": round(float(cosine[i]), 4),
                }
            )

        scored.sort(key=lambda x: (-x["score"], -x["bm25"], x["doc_id"]))
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
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    "SELECT collection, COUNT(*) as cnt FROM rag_documents GROUP BY collection"
                ).fetchall()
            finally:
                conn.close()
        counts = {c: 0 for c in self.COLLECTIONS}
        for r in rows:
            counts[r["collection"]] = int(r["cnt"])
        return counts
