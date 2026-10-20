# Data_Talk Enterprise BI (`v2.4`)

**Governed Conversational Business Intelligence, Semantic Layer Studio & Bidirectional DAX $\leftrightarrow$ SQL Compiler**

Data_Talk is a production-ready, zero-login enterprise conversational BI platform that connects to **CSV / Excel / Parquet files**, **SQL Databases** (`DuckDB`, `PostgreSQL`, `MySQL`, `SQLite`, `SQL Server`), and **Power BI Semantic Models** (`.pbix` schemas, Power BI Desktop Localhost XMLA, and Power BI Cloud XMLA).

Unlike generic "Text-to-SQL" wrappers that hallucinate on massive or cryptic schemas, Data_Talk builds a **Two-Tier Governed Semantic Brain** before answering executive queries:
1. **Tier 1 — Organizational Knowledge Framework (`okf.yaml` + `NetworkX` Join Graph)**: Deterministic schema metadata, Fact vs. Dimension classification, verified join paths, and governed DAX/SQL measures.
2. **Tier 2 — Hybrid Vector + BM25 RAG (`rag_memory.sqlite`)**: Local SQLite-backed TF-IDF + BM25 + Cosine Hybrid Retriever that indexes table/column definitions, business interview rules, and verified DAX/SQL few-shot examples with **Reciprocal Rank Fusion (RRF)**.

---

## ✨ Key Product Capabilities

- **01. Multi-Source Connector & Push-Down Statistical Profiler**
  - Zero-row-load push-down profiling (`COUNT(*)`, `COUNT(DISTINCT col)`, `null_ratio`, `min`/`max`, and PII-redacted sample values).
  - Automatic **Anomaly & Data Quality Watchdog** (detects negative revenues, null spikes, and cancelled order flags like `ord_stat = 'X'`).
- **02. Interactive SVG ER Diagram & 3-Signal Join Inference**
  - Combines **Database FK constraints**, **Canonical Name Similarity**, and **Value Overlap Inclusion Dependency** to infer $N:1$, $1:N$, and $1:1$ relationships.
  - Interactive Star/Snowflake ER graph with 1-hop focal filtering and human-in-the-loop join approval.
- **03. Smart AI Business Knowledge Interviewer**
  - Automatically detects cryptic columns (`ord_stat`, `rev_net_usd`, `cust_tier_cd`), generates high-impact domain questions with clickable 1-click answer chips, and indexes answers into both `okf.yaml` and Hybrid RAG.
- **04. Unified DAX $\leftrightarrow$ SQL Measure & Calculated Column Studio**
  - Bidirectional compiler translating Power BI DAX (`SUM`, `AVERAGE`, `DISTINCTCOUNT`, `DIVIDE`, `CALCULATE`) into dialect-aware SQL (`FILTER (WHERE ...)` / `NULLIF`) and compiling SQL metrics back into `DEFINE MEASURE` / `EVALUATE SUMMARIZECOLUMNS(...)` DAX queries.
- **05. Conversational BI, Root-Cause Variance Decomposition & Multi-Fact Fan-Trap Guard**
  - Automatically detects when a query spans multiple Fact tables (e.g., `orders` + `returns`) and compiles **separate pre-aggregated CTEs** joined on shared dimensions to prevent row multiplication (Chasm/Fan traps).
  - Includes a **"Why Did It Change?" Root-Cause Analyzer** with Period-over-Period Waterfall charts, **"Explain This Number"** SQL/DAX lineage inspector, **1-Click "Teach AI" feedback loop**, and **CSV Export**.
- **06. Executive Pinboard, What-If Scenario Simulator & Boardroom Report Export**
  - Live interactive Revenue & COGS shock sliders (`-30%` to `+50%`), pinned dashboard widgets, **Command Palette (`Ctrl+K`)**, **Live Read-Only SQL Scratchpad**, and 1-click **Executive BI Report (`.html` / Print-to-PDF)**.
- **6-Layer Enterprise Security & Zero-Leak Privacy Firewall**
  - **Layer 1**: Automatic PII column detection & `<REDACTED_PII_COLUMN>` masking.
  - **Layer 2**: Prompt-injection isolation via `<untrusted_data_sample>` tags.
  - **Layer 3**: `sqlglot` AST Read-Only Firewall (blocks `DROP`, `DELETE`, `UPDATE`, `INSERT`, `COPY`, `ATTACH`, and file-read functions).
  - **Layer 4**: DAX `EVALUATE` Read-Only Sandbox Guard.
  - **Layer 5**: Fernet-encrypted local credential vault (`.secrets/vault.enc`).
  - **Layer 6**: Outbound LLM Prompt Scrubber & Live Audit Log with 3 Privacy Modes (`Strict Zero-Data-Leak`, `Sanitized Samples`, `100% Local Air-Gapped`).

---

## 🚀 Quickstart

### 1. Local Python Execution

```bash
# Install dependencies
pip install -r requirements.txt

# Copy environment template and configure your Groq / NVIDIA NIM API keys
cp .env.example .env

# Start the production FastAPI + Web Studio server on http://127.0.0.1:8504
python -m uvicorn server:app --host 127.0.0.1 --port 8504
```

Open **`http://127.0.0.1:8504`** in your browser.

### 2. Docker Deployment

```bash
docker compose up --build -d
```

---

## 🔌 REST API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/api/health` | Production health check, DuckDB version, and active LLM provider readiness |
| `GET` | `/api/state` | Full workspace state, Semantic Readiness Score (`0-100%`), OKF, and RAG telemetry |
| `POST` | `/api/connect/demo` | Loads & profiles the 4-table enterprise demo warehouse (`orders`, `customers`, `products`, `returns`) |
| `POST` | `/api/connect/upload` | Ingests & profiles uploaded `.csv`, `.xlsx`, `.parquet`, or `.pbix` files |
| `POST` | `/api/connect/sql` | Connects to SQL databases (`duckdb`, `postgres`, `mysql`, `sqlite`, `tsql`) |
| `POST` | `/api/connect/powerbi` | Connects to Power BI Desktop localhost XMLA or Power BI Cloud XMLA |
| `POST` | `/api/joins/update` | Updates relationship governance (`approved`, `pending`, `rejected`, cardinality, join type) |
| `POST` | `/api/interview/answer` | Saves business interview answers into `okf.yaml` and Hybrid Vector + BM25 RAG |
| `POST` | `/api/measures/create` | Compiles, verifies, and persists bidirectional DAX $\leftrightarrow$ SQL measures |
| `POST` | `/api/chat/query` | Executes governed conversational BI query with self-healing retry loop |
| `POST` | `/api/sql/execute` | Executes a read-only SQL query from the Live SQL Scratchpad |
| `GET` | `/api/export/report` | Generates a printable Executive BI Report (`.html` / Print-to-PDF) |
| `GET` | `/api/export/okf` | Downloads the governed `okf.yaml` semantic layer |
| `GET` | `/api/export/dax` | Downloads the Power BI `.dax` measures script |

---

## 🧪 Running the Automated Test Suite

```bash
python -m pytest -v
```
