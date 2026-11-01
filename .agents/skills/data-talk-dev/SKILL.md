---
name: data-talk-dev
description: Complete development context, architecture blueprint, session history, and debugging playbook for Data_Talk Conversational BI. Use this skill when continuing development, reviewing code changes, debugging query compilation, or onboarding an agent on a new device.
---

# Data_Talk Development Context & Agent Playbook

> **Repository**: `Data_Talk`  
> **Active Branch**: `feature/conversational-ux-business-intelligence`  
> **Engine**: Python 3.11+ / FastAPI / DuckDB / LiteLLM / Plotly / Vanilla ES6 Modular Frontend  
> **Test Suite**: 82 Automated Tests (`python -m pytest tests/ -q`) — 100% Passing

---

## 1. Project Mission & Architecture Overview

Data_Talk is a governed, zero-login enterprise conversational BI platform that connects to **multiple simultaneous SQL databases** (PostgreSQL, MySQL, SQLite, DuckDB, SQL Server), **CSV/Excel/Parquet files**, and **Power BI Semantic Models** (`.pbix` files, Power BI XMLA).

### Two-Tier Governed Semantic Brain
1. **Tier 1 — Organizational Knowledge Framework (`OKFStore`, `okf.yaml`, `NetworkX` Join Graph)**:
   - Stores table metadata, column profiles, primary/foreign keys, cardinality, and calculated measures.
   - Computes deterministic Steiner-tree join paths and flags multi-fact chasm/fan traps.
   - Heals table roles (`FACT` vs. `DIMENSION`) and infers compound foreign keys (`degree_programs.program_id`, `course_sections.section_id`, `academic_terms.term_id`).
2. **Tier 2 — Hybrid Vector + BM25 RAG (`RAGStore`, `rag_memory.sqlite`)**:
   - SQLite-backed BM25 lexical retriever combined with BLAKE2b feature hashing and cosine vector similarity.
   - Fused using Reciprocal Rank Fusion ($k=60$) across schema glossaries, business rules, and verified few-shot queries.

### Two-Path Query Compiler (`QueryAgent`)
- **Path A — LLM Semantic Compiler (`_llm_compile_query`)**: Invokes LiteLLM (`LLMRouter`) when an API key (Groq, NVIDIA NIM, OpenAI, Gemini, Anthropic, or Ollama) is configured.
- **Path B — Offline Deterministic Heuristic Compiler (`_deterministic_sql_compiler`)**: 100% offline rule-based semantic SQL compiler. Handles multi-table joins, conformed dimension grouping, time series, and multi-fact CTEs without needing an LLM key.
- **Execution Mode Tracking**: Every query response carries `compiler_mode` (`"llm"` or `"heuristic"`) and `compiler_detail`, displayed as an interactive badge in the UI.

### Security & Privacy Firewalls
- **`SQLFirewall` (`src/security/sql_firewall.py`)**: `sqlglot` AST parser ensuring read-only queries. Blocks mutating commands (`DROP`, `DELETE`, `UPDATE`, `INSERT`), file read exploits (`read_csv`, `read_parquet`, `glob`), and blocked columns.
- **`PrivacyFirewall` (`src/security/privacy_firewall.py`)**: Enforces strict schema-only or PII-redacted sampling modes before prompts reach any LLM.

---

## 2. Chronological Conversation & Development History

This section captures every problem addressed, architectural decision made, and bug resolved across our development sessions:

### Session 1: Ingestion Feedback & Loading Overlay
- **Problem**: Ingesting large Excel/CSV datasets appeared frozen because the UI lacked feedback while the backend parsed sheets, profiled columns, and computed join graphs.
- **Resolution**: Created a real-time modal progress stepper in `frontend/app.js` with animated status transitions:
  1. `Reading and parsing file sheets...`
  2. `Profiling statistical distributions and PII...`
  3. `Inferring Star/Snowflake relationship graph...`
  4. `Persisting OKF semantic catalog & vector memory...`

### Session 2: Out-of-Domain Guardrail
- **Problem**: When asked questions about data not present in the active workspace, generic text-to-SQL agents hallucinate or run arbitrary SQL on unrelated tables.
- **Resolution**: Added an out-of-domain detection check in `QueryAgent.answer_question`. If zero tables, columns, or measures match the query intent, the agent gracefully returns an advisory card listing available tables in the workspace and advising how to connect new data.

### Session 3: Drill-Down Dropdowns & Chart Click Interactivity
- **Problem**: The drill-down action button was unresponsive, and questions arose regarding why drill-downs execute a query rather than doing an in-memory client-side filter.
- **Resolution**:
  - Fixed DOM event listeners for `.btn-drilldown-toggle` and `.drilldown-opt-btn` in `frontend/app.js`.
  - Added contextual conformed dimension options (`resp.drill_down_dimensions`) derived from the OKF join graph.
  - Linked Plotly click events (`plotly_click`) to a floating drill banner allowing 1-click drill-downs into clicked categorical values.
  - Retained SQL-level re-querying so drill-downs compute accurate database-level aggregations over hundreds of thousands of rows without requiring massive datasets to be shipped to the client browser.

### Session 4: Removing Artificial Row Limits
- **Problem**: Users noticed queries were truncated with `LIMIT 3` or `LIMIT 10`, omitting valid entities (e.g., only 3 colleges shown when 8 colleges existed in the dataset).
- **Resolution**: Removed hardcoded `LIMIT` clauses from analytical grouped queries in `QueryAgent._deterministic_sql_compiler`. Artificial limits now only apply when the user explicitly requests them (e.g., `"top 5"`, `"bottom 3"`, `"limit 10"`).

### Session 5: 4-Subagent Multi-Persona Audit & Bug Hunt
A comprehensive audit was performed using four specialized AI personas:
1. **Senior Data Analyst**:
   - *Fixed*: Non-additive metrics (`avg_salary`, `gpa`, `rate`, `pct`, `margin`) previously used `barmode="stack"`, adding averages together on the Y-axis. Updated `ChartRecommender` to never stack non-additive metrics and recommend multi-series `line` charts for temporal breakdowns.
   - *Fixed*: Time-series executive summaries previously reported `"Top Performer"` and `"% of total"` across calendar years. Replaced with **Trend Trajectory** (start-to-end change & %), **Peak/Lowest Period**, and **Period Benchmark**.
2. **Business Data Analyst**:
   - *Fixed*: Highest expenses or costs were labeled `"🏆 Top Performer"`. Added cost-aware badges (`💰 Highest Expenditure / Cost`).
   - *Fixed*: Neutral demographic splits (e.g. gender ratios) now use `📊 Highest Representation` rather than value judgments.
   - *Fixed*: Added multi-metric comparison summaries (`total_earnings` vs. `total_expenditure`, net surplus/deficit).
3. **Power BI & Semantic Expert**:
   - *Fixed*: Inferred compound table PK/FK relationships (`degree_programs.program_id`, `course_sections.section_id`, `academic_terms.term_id`).
   - *Fixed*: Automatic role healing on workspace load so fact tables (`enrollments_marks`, `fee_invoices`, `course_sections`) are classified as `FACT` and master tables (`colleges`, `departments`, `degree_programs`, `faculty`, `students`) are classified as `DIMENSION`.
4. **Code Logic Bug Hunter**:
   - *Fixed*: The query `"Show faculty salary for each college"` previously extracted `each` and generated `WHERE LOWER(colleges.college_name) LIKE '%each%'`. Added stop-words to prevent filter injection.
   - *Fixed*: `"Bottom 3"` / `"lowest"` queries now sort `ASC` instead of `DESC`.
   - *Fixed*: Single-table categorical queries now group by user-specified dimensions (e.g. `academic_rank`) rather than arbitrary first columns like `first_name`.

### Session 6: Overview Follow-Ups & Cover-Sheet Filtering
- **Problem**: In `"what is the data about"`, cover sheets (Excel tabs like `overview` containing `Unnamed_*` columns) were listed as entities, and follow-up chips suggested `"Drill down into enrollments_marks"`.
- **Resolution**:
  - Filtered out spreadsheet cover sheets ($\ge 50\%$ `Unnamed_*` columns) and isolated tables from `_generate_dataset_overview`.
  - Replaced follow-up overwrites with clean business questions (`"Show average marks for each college"`, `"Show total fee due by college"`, etc.).

### Session 7: Typo Tolerance & Multi-Fact Comparison Engine
- **Problem 1**: A 1-letter typo (`"show me total cost of each deparement year on year"`) failed exact string matching, causing fallback to `students` where `date_of_birth` was aliased to `year_of_birth` and `current_semester_num` to `total_current_semester_num`.
- **Problem 2**: In `"show me a comparison between the expenditure and earning of each college"`, missing keywords caused fallback to `colleges` where all rows showed `row_count = 1.0` grouped by `college_code`.
- **Resolution**:
  - Added `_fuzzy_matches_word` (using `difflib.SequenceMatcher`, threshold 0.80) in `_detect_relevant_tables` and dimension extraction.
  - Guarded against `date_of_birth` ever being used as a business trend date and blocked `current_semester_num` from financial sums.
  - Implemented **Case B2 (Multi-Fact Comparison CTE)** in `_deterministic_sql_compiler`:
    ```sql
    WITH dept_budget_cte AS (
        SELECT college_id, SUM(budget_allocated) AS department_budget FROM departments GROUP BY college_id
    ),
    faculty_payroll_cte AS (
        SELECT d.college_id, SUM(f.salary) AS faculty_payroll FROM faculty f JOIN departments d ON f.department_id = d.department_id GROUP BY d.college_id
    ),
    fee_earnings_cte AS (
        SELECT s.college_id, SUM(fi.amount_paid) AS total_earnings, SUM(fi.net_amount_due) AS total_fee_billed FROM fee_invoices fi JOIN students s ON fi.student_id = s.student_id GROUP BY s.college_id
    )
    SELECT c.college_name AS college_name,
           ROUND(COALESCE(er.total_earnings, 0), 2) AS total_earnings,
           ROUND(COALESCE(db.department_budget, 0) + COALESCE(fp.faculty_payroll, 0), 2) AS total_expenditure,
           ROUND(COALESCE(db.department_budget, 0), 2) AS department_budget,
           ROUND(COALESCE(fp.faculty_payroll, 0), 2) AS faculty_payroll,
           ROUND(COALESCE(er.total_earnings, 0) - (COALESCE(db.department_budget, 0) + COALESCE(fp.faculty_payroll, 0)), 2) AS net_surplus_deficit
    FROM colleges c
    LEFT JOIN dept_budget_cte db ON c.college_id = db.college_id
    LEFT JOIN faculty_payroll_cte fp ON c.college_id = fp.college_id
    LEFT JOIN fee_earnings_cte er ON c.college_id = er.college_id
    ORDER BY total_earnings DESC;
    ```
  - Updated `ChartRecommender` to plot `total_earnings` and `total_expenditure` side-by-side in a **Grouped Bar Chart** (`bar`).
  - Corrected single-table categorical fallback so `college_name` is selected ahead of `college_code`.
  - In 2D time-series queries (`department_name` + `hire_year`), swapped axes so temporal columns are always on the X-axis, recommended `"line"`, and capped series to Top 10 departments for visual clarity.

### Session 8: LLM vs. Heuristic Engine Execution Indicator
- **Problem**: Users could not tell whether a query was solved via an active LLM provider or via the built-in deterministic heuristic compiler.
- **Resolution**:
  - Added `compiler_mode` and `compiler_detail` to `QueryResponse`.
  - Added visual indicator badges in the UI:
    - 🟢 `✨ LLM (<model>)` (emerald) when generated by the LLM.
    - 🟠 `⚙️ Heuristic Engine` (amber, with hover tooltip explaining reason, e.g. `No LLM Key` or `Heuristic Fallback`) when compiled via offline heuristic rules.
  - Displayed the compiler engine details in the `<> Explain` lineage drawer.

### Session 9: Verified Query Memory (Golden Query Cache) & Studio Tab
- **Concept**: User-verified conversational query caching ("Golden Queries"). When an analyst verifies that generated SQL/DAX code is canonical, they can bookmark it directly into Workspace memory.
- **Runtime Resolution**:
  - `QueryAgent` performs pre-compilation intent matching (`_find_matching_verified_query`) against `workspace.verified_queries` using normalized exact string matching combined with token Jaccard similarity.
  - On hit, bypasses the LLM/Heuristic compiler and runs the canonical recipe directly on the live database connector (sub-20ms live data refresh).
  - Emits `compiler_mode = "verified_memory"` with `⚡ Verified Memory` badge and provides a `Re-run with AI` escape hatch (`force_recompile=True`).
  - Added full REST API in `src/api/routes/verified_queries.py`: List, Create, Edit, Delete, and Test execution.
  - Added dedicated **Query Memory** tab in frontend sidebar with search, test run preview, edit modal (modifying question trigger pattern, SQL, DAX, or chart type), and delete action.

---

## 3. Key Invariants & Rules for AI Agents Developing on This Repo

When working on this repository, you **MUST** follow these core guidelines:

1. **Git Branch Discipline**:
   - Always stay on `feature/conversational-ux-business-intelligence`.
   - **NEVER touch or push directly to `main`**.
2. **Never Break Automated Tests**:
   - Always verify changes with `python -m pytest tests/ -q`.
   - All 82 tests must pass before committing.
3. **No Artificial `LIMIT` Clauses**:
   - Grouped analytical queries must never inject artificial `LIMIT` clauses (like `LIMIT 3`) unless explicitly specified in the query prompt.
4. **Non-Additive Metric Guardrail**:
   - Never stack averages, rates, GPAs, or margins (`barmode="stack"`).
   - Never sum averages in executive summaries or calculate `share = val / sum(avgs)`.
5. **Fan & Chasm Trap Guardrail**:
   - Never join multiple Fact tables (e.g. `orders` + `returns`, or `fee_invoices` + `departments`) in a single flat `JOIN`. Always use pre-aggregated CTEs on conformed dimension keys.
6. **Date & Metric Guardrails**:
   - `date_of_birth` / `dob` must never be picked as a business trend date column unless the user specifically asks for birth or age analysis.
   - `current_semester_num`, `credit_hours`, and section numbers must never be summed as financial metrics.
7. **Human-Readable Name Priority**:
   - Always prefer `*_name` or `title` columns over `*_code` or `*_id` columns when selecting grouping dimensions.

---

## 4. Key Directory & File References

- **FastAPI Server Entrypoint**: [`server.py`](file:///c:/Users/pashi/Desktop/Data_Talk/server.py)
- **Conversational BI Query Agent**: [`src/agents/query_agent.py`](file:///c:/Users/pashi/Desktop/Data_Talk/src/agents/query_agent.py)
- **LiteLLM Multi-Provider Router**: [`src/agents/llm_router.py`](file:///c:/Users/pashi/Desktop/Data_Talk/src/agents/llm_router.py)
- **Plotly Visual Recommender**: [`src/viz/chart_recommender.py`](file:///c:/Users/pashi/Desktop/Data_Talk/src/viz/chart_recommender.py)
- **OKF Semantic Store & Join Engine**: [`src/knowledge/okf_store.py`](file:///c:/Users/pashi/Desktop/Data_Talk/src/knowledge/okf_store.py)
- **Join Inference Engine**: [`src/knowledge/join_inference.py`](file:///c:/Users/pashi/Desktop/Data_Talk/src/knowledge/join_inference.py)
- **Hybrid Vector + BM25 RAG Store**: [`src/knowledge/rag_store.py`](file:///c:/Users/pashi/Desktop/Data_Talk/src/knowledge/rag_store.py)
- **AST SQL Firewall**: [`src/security/sql_firewall.py`](file:///c:/Users/pashi/Desktop/Data_Talk/src/security/sql_firewall.py)
- **Privacy Firewall**: [`src/security/privacy_firewall.py`](file:///c:/Users/pashi/Desktop/Data_Talk/src/security/privacy_firewall.py)
- **Frontend SPA (Vanilla ES6)**: [`frontend/app.js`](file:///c:/Users/pashi/Desktop/Data_Talk/frontend/app.js) & [`frontend/index.html`](file:///c:/Users/pashi/Desktop/Data_Talk/frontend/index.html)
- **Automated Test Suite**: [`tests/`](file:///c:/Users/pashi/Desktop/Data_Talk/tests/)
