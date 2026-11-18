"""Shared semantic text normalization, categorical value matching, measure matching, and table detection."""

from __future__ import annotations

import difflib
import re
from typing import TYPE_CHECKING, Any

from src.knowledge.okf_models import OKFWorkspace, SemanticMeasure
from src.security.privacy_firewall import PrivacyFirewall

if TYPE_CHECKING:
    from src.connectors.base import BaseConnector


def normalize_what_if_overrides(overrides: dict[str, float] | None) -> dict[str, float]:
    """Strip zero-value What-If levers so inactive UI defaults never suppress prompt shock parsing."""
    if not overrides:
        return {}
    cleaned: dict[str, float] = {}
    for k, v in overrides.items():
        try:
            fval = float(v)
        except (TypeError, ValueError):
            continue
        if abs(fval) > 1e-9:
            cleaned[k] = fval
    return cleaned


def extract_what_if_from_prompt(question: str) -> dict[str, float]:
    """Extract conversational What-If multipliers from user question (e.g. 'what if sales increase by 10%')."""
    q = question.lower()
    if not any(
        k in q
        for k in (
            "what if",
            "simulate",
            "what-if",
            "shock",
            "suppose",
            "if cost",
            "if sales",
            "if price",
            "if tuition",
            "if budget",
            "inflation",
        )
    ):
        return {}

    overrides: dict[str, float] = {}

    m_rev = re.search(r"(?:revenue|sales|price|fee|tuition|income)([^0-9\n]{1,40}?)([+-]?\d+(?:\.\d+)?)\s*%", q)
    if not m_rev:
        m_rev_rev = re.search(
            r"([+-]?\d+(?:\.\d+)?)\s*%\s*(?:[^0-9\n]{0,20}?)(?:revenue|sales|price|fee|tuition|income)",
            q,
        )
        if m_rev_rev:
            overrides["Revenue Multiplier %"] = float(m_rev_rev.group(1)) / 100.0
    else:
        val = float(m_rev.group(2))
        mid = m_rev.group(1)
        if any(w in mid for w in ("drop", "down", "dec", "fall", "cut", "-")):
            val = -abs(val)
        overrides["Revenue Multiplier %"] = val / 100.0

    m_cost = re.search(r"(?:cost|cogs|expense|expenditure|salary|budget|inflation)([^0-9\n]{1,40}?)([+-]?\d+(?:\.\d+)?)\s*%", q)
    if not m_cost:
        m_cost_rev = re.search(
            r"([+-]?\d+(?:\.\d+)?)\s*%\s*(?:[^0-9\n]{0,20}?)(?:cost|cogs|expense|expenditure|salary|budget|inflation)",
            q,
        )
        if m_cost_rev:
            overrides["Cost Multiplier %"] = float(m_cost_rev.group(1)) / 100.0
    else:
        val = float(m_cost.group(2))
        mid = m_cost.group(1)
        if any(w in mid for w in ("drop", "down", "dec", "fall", "cut", "-")):
            val = -abs(val)
        overrides["Cost Multiplier %"] = val / 100.0

    return normalize_what_if_overrides(overrides)


def normalize_prompt_text(text: str) -> str:
    """Sanitize Unicode non-breaking hyphens, smart quotes, and noisy parentheticals."""
    if not text:
        return ""
    cleaned = re.sub(r"[\u2010\u2011\u2012\u2013\u2014\u2015\u2212]", "-", text)
    cleaned = re.sub(r"[\u2018\u2019]", "'", cleaned)
    cleaned = re.sub(r"[\u201c\u201d]", '"', cleaned)
    cleaned = re.sub(r"\(\s*e\.g\.[^)]*\)", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def fuzzy_matches_word(q_words: list[str], candidate: str, threshold: float = 0.80) -> bool:
    """Return True if any token in q_words matches candidate exactly or with fuzzy similarity >= threshold."""
    cand = candidate.lower().strip()
    if not cand:
        return False
    if cand in q_words:
        return True
    if len(cand) < 5:
        return False
    for qw in q_words:
        if len(qw) >= 5 and abs(len(qw) - len(cand)) <= 2:
            if difflib.SequenceMatcher(None, qw, cand).ratio() >= threshold:
                return True
    return False


def ensure_categorical_samples(
    workspace: OKFWorkspace,
    connector: "BaseConnector | None" = None,
) -> None:
    """Populate sample_values in-memory once per run for medium-cardinality categorical columns."""
    if connector is None or getattr(connector, "dialect", "") == "dax":
        return
    skip_cols = {
        "first_name",
        "last_name",
        "email",
        "phone",
        "phone_number",
        "roll_number",
        "address",
        "remarks",
        "syllabus_summary",
        "date_of_birth",
    }
    for tbl in workspace.active_tables():
        for col in tbl.columns:
            if col.is_blocked or col.is_pii or col.is_primary_key or col.is_foreign_key:
                continue
            c_low = col.name.lower()
            if c_low in skip_cols or c_low.endswith("_id"):
                continue
            if not any(ty in col.data_type.upper() for ty in ("CHAR", "TEXT", "VARCHAR", "STRING")):
                continue
            if 0 < col.distinct_count <= 100 and len(col.sample_values) < min(col.distinct_count, 50):
                if col.sample_values == ["<REDACTED_PII_COLUMN>"]:
                    continue
                try:
                    samp_sql = (
                        f'SELECT DISTINCT CAST("{col.name}" AS VARCHAR) AS val '
                        f'FROM "{tbl.name}" WHERE "{col.name}" IS NOT NULL LIMIT 65'
                    )
                    samp_df = connector.execute_query(samp_sql, max_rows=65)
                    if not samp_df.empty and "val" in samp_df.columns:
                        raw_vals = [str(v) for v in samp_df["val"].tolist() if v is not None]
                        sanitized, pii_found = PrivacyFirewall.sanitize_profile_samples(
                            col.name, raw_vals, col.distinct_count
                        )
                        if not pii_found and sanitized:
                            col.sample_values = sanitized
                            if len(raw_vals) < 65:
                                col.distinct_count = len(sanitized)
                        elif pii_found:
                            col.sample_values = ["<REDACTED_PII_COLUMN>"]
                        elif samp_df.empty:
                            col.distinct_count = 0
                except Exception:
                    pass


def detect_categorical_value_matches(
    question: str,
    workspace: OKFWorkspace,
) -> list[dict[str, Any]]:
    """Detect partial or exact categorical column values mentioned in the user's natural-language question."""
    q_clean = normalize_prompt_text(question)
    q_low = q_clean.lower()
    q_words = [w for w in re.findall(r"[a-z0-9]+", q_low) if len(w) >= 3]
    if not q_words:
        return []

    quoted_segments = [
        m.strip().lower()
        for m in re.findall(r"['\"]([^'\"]{2,80})['\"]", q_clean)
        if m.strip()
    ]

    generic_single_tokens = {
        "show", "list", "give", "tell", "what", "which", "where", "when", "who", "how",
        "many", "much", "total", "sum", "avg", "average", "mean", "count", "number",
        "ratio", "ritio", "raio", "rio", "share", "percent", "percentage", "trend",
        "monthly", "yearly", "annual", "year", "years", "month", "months", "date", "time",
        "top", "bottom", "highest", "lowest", "best", "worst", "most", "least",
        "all", "each", "every", "per", "by", "for", "in", "of", "from", "to", "with",
        "on", "at", "and", "or", "the", "this", "that", "these", "those", "it", "them",
        "break", "broken", "down", "drill", "into", "split", "across", "wise", "compare",
        "versus", "between", "filter", "filtered", "only", "just", "over", "under", "more", "less",
        "table", "tables", "column", "columns", "data", "dataset", "overview", "summary",
        "college", "colleges", "department", "departments", "dept", "student", "students",
        "faculty", "professor", "professors", "teacher", "instructor", "course", "courses",
        "program", "programs", "degree", "degrees", "level", "rank", "gender", "status",
        "fee", "fees", "invoice", "invoices", "tuition", "salary", "salaries", "payroll",
        "budget", "cost", "costs", "expense", "expenses", "expenditure", "expenditures",
        "earning", "earnings", "revenue", "revenues", "sales", "profit", "margin",
        "order", "orders", "return", "returns", "customer", "customers", "product", "products",
        "category", "categories", "region", "regions", "tier", "tiers", "channel", "channels",
        "mark", "marks", "grade", "grades", "score", "scores", "gpa", "assessment", "attendance",
        "science", "sciences", "studies", "development", "management", "system", "systems",
        "general", "applied", "arts", "bachelor", "master", "doctor", "associate", "assistant",
        "paid", "due", "outstanding", "balance", "net", "gross", "credit", "credits", "active",
    }
    val_stop_words = {"of", "and", "the", "in", "for", "to", "a", "an", "&"}

    by_tail_match = re.findall(
        r"\b(?:by|per|for\s+each|each|broken\s+down\s+by|split\s+by)\s+([a-z0-9_]+(?:\s+[a-z0-9_]+)?)",
        q_low,
    )
    grouping_phrases = " ".join(by_tail_match)

    candidates: list[dict[str, Any]] = []
    for tbl in workspace.active_tables():
        for col in tbl.columns:
            if col.is_blocked or col.is_pii or col.is_primary_key or col.is_foreign_key:
                continue
            c_low = col.name.lower()
            if c_low.endswith("_id") or c_low.endswith("_code") or not col.sample_values:
                continue
            if col.sample_values == ["<REDACTED_PII_COLUMN>"]:
                continue

            val_tokens_map: dict[str, list[str]] = {}
            token_freq: dict[str, int] = {}
            for raw_val in col.sample_values:
                v_str = str(raw_val).strip()
                if not v_str or len(v_str) < 2:
                    continue
                v_toks = [
                    t
                    for t in re.findall(r"[a-z0-9]+", v_str.lower())
                    if len(t) >= 3 and t not in val_stop_words
                ]
                val_tokens_map[v_str] = v_toks
                for vt in set(v_toks):
                    token_freq[vt] = token_freq.get(vt, 0) + 1

            for v_str, v_toks in val_tokens_map.items():
                v_low = v_str.lower()
                score = 0
                matched_phrase = ""

                if any(qs == v_low or (len(qs) >= 4 and qs in v_low) for qs in quoted_segments):
                    score = 150 + len(v_str)
                    matched_phrase = v_str
                elif len(v_low) >= 4 and re.search(rf"\b{re.escape(v_low)}\b", q_low):
                    if v_low in grouping_phrases and len(v_toks) == 1 and v_low in generic_single_tokens:
                        continue
                    score = 130 + len(v_str)
                    matched_phrase = v_str
                else:
                    matched_toks: list[str] = []
                    for vt in v_toks:
                        if vt in ("college", "department", "program", "bachelor", "master", "bs", "ms", "ba", "phd"):
                            continue
                        if vt in q_words:
                            matched_toks.append(vt)
                        elif len(vt) >= 6:
                            for qw in q_words:
                                if len(qw) >= 6 and abs(len(qw) - len(vt)) <= 2:
                                    if difflib.SequenceMatcher(None, qw, vt).ratio() >= 0.84:
                                        matched_toks.append(qw)
                                        break

                    if len(matched_toks) >= 2:
                        score = 105 + (12 * len(matched_toks))
                        matched_phrase = " ".join(matched_toks)
                    elif len(matched_toks) == 1:
                        mt = matched_toks[0]
                        if mt not in generic_single_tokens and len(mt) >= 4:
                            freq = token_freq.get(mt, 1)
                            if freq == 1:
                                score = 92
                                matched_phrase = mt
                            elif freq <= 2 and c_low.endswith("_name"):
                                score = 72
                                matched_phrase = mt

                if score > 0:
                    if c_low.endswith("_name"):
                        score += 4
                    candidates.append({
                        "table": tbl.name,
                        "column": col.name,
                        "value": v_str,
                        "score": score,
                        "matched_phrase": matched_phrase,
                    })

    candidates.sort(key=lambda x: (-x["score"], len(x["value"])))
    deduped: list[dict[str, Any]] = []
    seen_cols: set[tuple[str, str]] = set()
    seen_phrases: set[str] = set()
    for cand in candidates:
        tc = (cand["table"], cand["column"])
        mp = cand["matched_phrase"].lower()
        if tc in seen_cols or (mp and mp in seen_phrases):
            continue
        seen_cols.add(tc)
        if mp:
            seen_phrases.add(mp)
        deduped.append(cand)
    return deduped


def detect_relevant_tables(
    question: str,
    workspace: OKFWorkspace,
    rag_bundle: dict[str, list[dict[str, Any]]] | None = None,
) -> list[str]:
    """Infer all active workspace tables referenced directly, via column hints, or via categorical values."""
    rag_bundle = rag_bundle or {}
    q_low = question.lower()
    q_norm = re.sub(r"[_\s]+", " ", q_low)
    q_words = [w for w in re.findall(r"[a-z0-9]+", q_low) if len(w) >= 2]
    found: list[str] = []

    generic_col_names = {
        "year", "date", "month", "day", "id", "status", "type", "name",
        "code", "desc", "description", "value", "created_at", "updated_at",
        "total", "count", "num", "number",
    }
    generic_col_tokens = {
        "total", "count", "amount", "value", "number", "num", "code", "name",
        "type", "status", "date", "year", "month", "day", "rate", "pct",
        "percent", "score", "fee", "usd", "id", "net", "current", "max", "min", "avg", "per",
    }
    all_table_nouns: set[str] = set()
    for _t in workspace.active_tables():
        _tl = _t.name.lower()
        all_table_nouns.add(_tl)
        if _tl.endswith("s"):
            all_table_nouns.add(_tl[:-1])
        for _part in _tl.split("_"):
            if len(_part) >= 3:
                all_table_nouns.add(_part)
                if _part.endswith("s"):
                    all_table_nouns.add(_part[:-1])

    for tbl in workspace.active_tables():
        t_low = tbl.name.lower()
        t_norm = re.sub(r"[_\s]+", " ", t_low)
        singular = t_low[:-1] if t_low.endswith("s") else t_low
        singular_norm = singular.replace("_", " ")
        if (
            re.search(rf"\b{re.escape(t_low)}\b", q_low)
            or re.search(rf"\b{re.escape(singular)}\b", q_low)
            or re.search(rf"\b{re.escape(t_norm)}\b", q_norm)
            or re.search(rf"\b{re.escape(singular_norm)}\b", q_norm)
            or fuzzy_matches_word(q_words, t_low)
            or fuzzy_matches_word(q_words, singular)
        ):
            found.append(tbl.name)
            continue
        for col in tbl.columns:
            if col.is_blocked:
                continue
            c_low = col.name.lower()
            c_norm = re.sub(r"[_\s]+", " ", c_low)
            if c_low in generic_col_names:
                continue
            if re.search(rf"\b{re.escape(c_low)}\b", q_low) or re.search(rf"\b{re.escape(c_norm)}\b", q_norm):
                found.append(tbl.name)
                break
            if c_low.endswith("_id") or col.is_primary_key or col.is_foreign_key:
                continue
            col_words = [w for w in c_norm.split() if len(w) > 3]
            if len(col_words) >= 2 and all(re.search(rf"\b{re.escape(w)}\b", q_norm) for w in col_words):
                found.append(tbl.name)
                break
            distinctive_col_words = [
                w
                for w in col_words
                if w not in generic_col_tokens and w not in all_table_nouns and len(w) >= 4
            ]
            if distinctive_col_words and any(
                re.search(rf"\b{re.escape(dw)}\b", q_norm) or fuzzy_matches_word(q_words, dw, 0.84)
                for dw in distinctive_col_words
            ):
                if tbl.name not in found:
                    found.append(tbl.name)
                break

    for cat_m in detect_categorical_value_matches(question, workspace):
        t_cat = cat_m["table"]
        if t_cat not in found and workspace.get_table(t_cat):
            found.append(t_cat)

    keyword_table_hints = {
        "customers": ("customer", "customers", "client", "clients", "tier", "region", "company", "ent", "smb"),
        "products": ("product", "products", "sku", "category", "categories", "hardware", "cloud", "suite"),
        "orders": ("order", "orders", "revenue", "sales", "profit", "margin", "discount", "channel", "qty", "month", "year", "top"),
        "returns": ("return", "returns", "refund", "refunds", "defective", "reason"),
        "enrollments_marks": ("grade", "grades", "mark", "marks", "score", "scores", "gpa", "exam", "assessment", "attendance", "enrollment", "enrol"),
        "students": ("student", "students", "gender", "dob", "birth", "admission", "semester"),
        "colleges": ("college", "colleges", "campus", "dean"),
        "courses": ("course", "courses", "credit", "credits", "syllabus"),
        "degree_programs": ("program", "degree", "major"),
        "departments": ("department", "departments", "dept", "budget", "cost", "costs", "expense", "expenses", "expenditure", "expenditures", "spend", "spending", "allocated", "outflow"),
        "fee_invoices": ("fee", "fees", "invoice", "invoices", "tuition", "paid", "balance", "due", "payment", "earning", "earnings", "income", "revenue", "revenues", "collection", "collections", "billed", "billing", "receipts"),
        "faculty": ("faculty", "professor", "professors", "teacher", "instructor", "salary", "salaries", "rank", "payroll", "wage", "wages", "compensation", "expenditure", "expenditures", "cost", "costs", "expense", "expenses"),
    }
    for t_name, kw_tuple in keyword_table_hints.items():
        if workspace.get_table(t_name) and any(
            re.search(rf"\b{re.escape(k)}\b", q_low) or fuzzy_matches_word(q_words, k)
            for k in kw_tuple
        ):
            if t_name not in found:
                found.append(t_name)

    if (
        "departments" in found
        and workspace.get_table("faculty")
        and "faculty" not in found
        and any(w in q_low for w in ("year", "trend", "annual", "yearly", "over time", "yoy"))
        and any(w in q_low for w in ("cost", "expense", "spend", "salary", "payroll", "budget"))
    ):
        found.append("faculty")

    if not found:
        for item in rag_bundle.get("schema_glossary", [])[:2]:
            if item.get("bm25", 0.0) >= 0.5 or item.get("cosine", 0.0) >= 0.35:
                t_meta = item.get("metadata", {}).get("table_name")
                if t_meta and workspace.get_table(t_meta) and t_meta not in found:
                    found.append(t_meta)

    return found


def match_measures(question: str, workspace: OKFWorkspace) -> list[SemanticMeasure]:
    """Match natural-language metric mentions against workspace SemanticMeasure definitions."""
    q_low = question.lower()
    matched: list[SemanticMeasure] = []
    stop_measure_words = {
        "total", "average", "count", "number", "amount", "value",
        "metric", "ratio", "share", "trend", "breakdown", "gross",
        "annual", "monthly", "overall", "summary",
    }
    for m in workspace.measures:
        m_low = m.name.lower()
        if re.search(rf"\b{re.escape(m_low)}\b", q_low) or any(
            re.search(rf"\b{re.escape(s.lower())}\b", q_low) for s in m.synonyms
        ):
            matched.append(m)
        elif any(
            re.search(rf"\b{re.escape(tok)}\b", q_low)
            for tok in m_low.split()
            if len(tok) > 4 and tok not in stop_measure_words
        ):
            matched.append(m)
    return matched
