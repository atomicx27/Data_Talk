# Data_Talk Enterprise BI (`v2.7.0`)

**Governed Conversational Business Intelligence, Semantic Layer Studio, Multi-Database Federation, Multi-Fact Comparison Engine & Bidirectional DAX $\leftrightarrow$ SQL Compiler**

Data_Talk is a production-ready, zero-login enterprise conversational BI platform that connects to **multiple simultaneous SQL databases** (`PostgreSQL`, `MySQL`, `SQLite`, `DuckDB`, `SQL Server`), **CSV / Excel / Parquet files**, and **Power BI Semantic Models** (`.pbix` schemas, Power BI Desktop Localhost XMLA, and Power BI Cloud XMLA).

Unlike generic "Text-to-SQL" wrappers that hallucinate on massive or cryptic schemas, Data_Talk builds a **Two-Tier Governed Semantic Brain** before answering executive queries:
1. **Tier 1 — Organizational Knowledge Framework (`okf.yaml` + `NetworkX` Join Graph)**: Deterministic schema metadata, Star Schema Fact vs. Dimension classification, verified multi-hop and cross-database join paths, compound foreign key resolution, and governed DAX/SQL measures.
2. **Tier 2 — Hybrid Vector + BM25 RAG (`rag_memory.sqlite`)**: Local SQLite-backed BM25 lexical retriever combined with stable BLAKE2b feature hashing and cosine vector similarity, fused using **Reciprocal Rank Fusion (RRF, $k=60$)**.

---

## ✨ Key Product Capabilities

- **01. Multi-Database Federation & Push-Down Statistical Profiler (`FederatedConnector`)**
  - Attach **multiple SQL databases concurrently** into a single unified workspace with isolated database schema aliasing (`"{source_id}"."{table}"`).
  - Zero-row-load push-down profiling (`COUNT(*)`, `COUNT(DISTINCT col)`, `null_ratio`, `min`/`max`, and PII-redacted sample values) running directly on remote source engines.
  - Automatic **Anomaly & Data Quality Watchdog** (detects negative revenues, null spikes, and cancelled order flags like `ord_stat = 'X'`).
- **02. Interactive SVG ER Diagram & Cross-Database Join Inference**
  - Discovers both intra-database and **cross-database relationships** (`🌐 Cross-DB`) using Database FKs, Canonical Name Similarity, and Value Overlap Probing.
  - Interactive Star/Snowflake ER graph with 1-hop focal filtering, database source badges, and human-in-the-loop join approval.
- **03. Smart AI Business Knowledge Interviewer & Cross-Database Link Clarifications**
  - Per-database question isolation and auto-generated cross-database join confirmation questions (e.g. *"Table sales_db.orders and crm_db.customers share customer_id. Should these databases be joined?"*).
  - High-impact domain questions with clickable 1-click answer chips indexed into both `okf.yaml` and Hybrid RAG.
- **04. Unified DAX $\leftrightarrow$ SQL Measure & Calculated Column Studio**
  - Bidirectional compiler translating Power BI DAX (`SUM`, `AVERAGE`, `DISTINCTCOUNT`, `DIVIDE`, `CALCULATE`) into dialect-aware SQL (`FILTER (WHERE ...)` / `NULLIF`) and compiling SQL metrics back into `DEFINE MEASURE` / `EVALUATE SUMMARIZECOLUMNS(...)` DAX queries.
- **05. Conversational BI, Root-Cause Variance Decomposition & Multi-Fact Fan-Trap Guard**
  - Automatically detects when a query spans multiple Fact tables (e.g., `orders` + `returns`) and compiles **separate pre-aggregated CTEs** joined on shared dimensions to prevent row multiplication (Chasm/Fan traps).
  - Multi-hop dimensional aggregation synthesis (`SELECT colleges.college_name, SUM(enrollments_marks.total_marks) ... GROUP BY colleges.college_name`) across multi-table relationships.
  - Includes a **"Why Did It Change?" Root-Cause Analyzer** with Period-over-Period Waterfall charts, **"Explain This Number"** SQL/DAX lineage inspector, **1-Click "Teach AI" feedback loop**, and **CSV Export**.
- **06. Executive Pinboard, What-If Scenario Simulator & Boardroom Report Export**
  - Live interactive Revenue & COGS shock sliders (`-30%` to `+50%`), pinned dashboard widgets, **Command Palette (`Ctrl+K`)**, **Live Read-Only SQL Scratchpad**, and 1-click **Executive BI Report (`.html` / Print-to-PDF)**.
- **07. Hardened Multi-Session Enterprise Security Architecture**
  - **Per-Session State Isolation**: Independent active workspaces, chat histories, what-if overrides, and LLM router configurations across concurrent browser tabs and users.
  - **Session Cookie Authentication & Origin Defense**: Signed, random `dt_session` cookies (`HttpOnly`, `SameSite=Strict`), strict CORS origin verification, and optional Bearer token mode via `DATA_TALK_API_TOKEN`.
  - **Strict Path Containment**: Path traversal rejection on workspace identifiers and uploaded filenames.
  - **3-Tier Credential Vault**: Master-key resolution order (`DATA_TALK_MASTER_KEY` env var $\rightarrow$ OS Keyring $\rightarrow$ atomic Fernet encrypted file) with decrypt verification.
  - **AST Read-Only Firewall**: AST query parser blocking `DROP`, `DELETE`, `UPDATE`, `INSERT`, `COPY`, `ATTACH`, file reading functions (`read_csv`, `read_parquet`, `glob`), and blocked column leakage.
- **08. Conversational UX, Multi-Persona Audit & Execution Transparency (`v2.7.0`)**
  - **Real-Time Ingestion Stepper**: Animated modal progress overlay tracking sheet reading, statistical profiling, relationship inference, and vector embedding persistence.
  - **Out-of-Domain Guardrail**: Graceful semantic advisory when questions reference missing domain entities, presenting active workspace catalogs without hallucinating arbitrary SQL.
  - **Interactive Conformed Drill-Downs**: Contextual conformed dimension menus and Plotly click banner integration for instant drilling without client-side memory limits.
  - **Multi-Persona Audit & BI Rigor**:
    - *Non-Additive Safeguards*: Non-additive metrics (`avg_salary`, `gpa`, `margin`, `rates`) never stack; multi-series Line Trend charts recommended for temporal breakdowns.
    - *Trend Trajectory Summaries*: Executive summaries report Start $\rightarrow$ End change %, Peak/Trough periods, and Period Benchmarks instead of summing calendar years.
    - *Neutral & Cost Framing*: Avoids calling highest expenditures or demographic splits "Top Performer", using `💰 Highest Cost` and `📊 Highest Representation`.
  - **Multi-Fact Comparison Engine**: Pre-aggregated CTE synthesis for comparing multi-fact expenditures vs. earnings across conformed dimensions (`colleges`, `departments`) without row multiplication or fan traps.
  - **Execution Mode Transparency**: Transparent live pill badges on every chat card (`✨ LLM (<model>)` vs. `⚙️ Heuristic Engine (<reason>)`) and `<> Explain` modal detailing whether queries were solved by an LLM or the deterministic heuristic compiler.
  - **Compound FK & Role Healing**: Automatic classification healing for `FACT` vs. `DIMENSION` roles and compound suffix foreign key inference (`program_id` $\rightarrow$ `degree_programs.program_id`, `section_id` $\rightarrow$ `course_sections.section_id`, `term_id` $\rightarrow$ `academic_terms.term_id`).
  - **Agent Onboarding Skill**: Packaged agent development context and debugging playbook in `.gemini/skills/data-talk-dev/SKILL.md` and `.agents/skills/data-talk-dev/SKILL.md`.

---

## 🚀 Quickstart

### 1. Local Python Execution

```bash
# Install production dependencies
pip install -r requirements.txt

# (Optional) Install development and testing dependencies
pip install -r requirements-dev.txt

# Copy environment template and configure your provider API keys
cp .env.example .env

# Start the production FastAPI server on http://127.0.0.1:8504
python server.py
```

Open **`http://127.0.0.1:8504`** in your browser.

### 2. Docker Deployment

```bash
docker compose up --build -d
```

### 3. Vercel Serverless Deployment

Data_Talk includes out-of-the-box support for Vercel deployment:
- **Root Entrypoint**: `api/index.py` routes serverless requests to the FastAPI ASGI application (`server:app`).
- **Configuration**: `vercel.json` configures the serverless Python runtime (`@vercel/python`) and dynamic routes.

To deploy via the Vercel CLI:
```bash
npm install -g vercel
vercel
```
Or import your GitHub repository directly in the [Vercel Dashboard](https://vercel.com/new).

> [!NOTE]
> Vercel Serverless functions run in read-only container filesystems with temporary `/tmp` storage. When deploying on Vercel, connect to remote SQL databases (PostgreSQL, MySQL, Neon, Supabase, PlanetScale) or use pre-loaded workspace configurations.

---

## 🔒 Security & Deployment Configuration

All security settings can be customized via `.env` or container environment variables:

| Variable | Default | Purpose |
| :--- | :--- | :--- |
| `DATA_TALK_API_TOKEN` | _None (Open Mode)_ | Require Bearer token authentication or browser token sign-in |
| `DATA_TALK_MASTER_KEY` | _Auto-generated_ | Master encryption key for the local credential vault |
| `DATA_TALK_ALLOWED_ORIGINS` | `http://127.0.0.1:8504,http://localhost:8504` | Allow-list for cross-origin API requests |
| `MAX_UPLOAD_MB` | `200` | Maximum file size for file uploads in megabytes |
| `PORT` | `8504` | Server listening port |
| `HOST` | `127.0.0.1` | Server listening interface (`0.0.0.0` in Docker) |

---

## 🔌 REST API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/api/health` | Production health check, DuckDB version, and provider readiness |
| `GET` | `/api/state` | Full session workspace state, Readiness Score (`0-100%`), and telemetry |
| `POST` | `/api/settings` | Updates session active workspace, LLM provider, privacy mode, and API keys |
| `POST` | `/api/connect/demo` | Loads & profiles the 4-table enterprise demo warehouse |
| `POST` | `/api/connect/upload` | Ingests & profiles uploaded `.csv`, `.xlsx`, `.parquet`, or `.pbix` files |
| `POST` | `/api/connect/sql` | Connects or attaches SQL databases (`duckdb`, `postgres`, `mysql`, `sqlite`, `tsql`) |
| `POST` | `/api/connect/remove-source` | Disconnects an attached SQL database, dropping its mirrored views and rules |
| `POST` | `/api/connect/powerbi` | Connects to Power BI Desktop XMLA or Power BI Cloud XMLA |
| `POST` | `/api/joins/approve-all` | Approves all candidate inferred relationships |
| `POST` | `/api/joins/update` | Updates relationship governance (cardinality, status, join type) |
| `POST` | `/api/joins/add` | Adds a custom user-defined relationship (supports cross-database links) |
| `POST` | `/api/interview/answer` | Saves business interview answers into `okf.yaml` and RAG memory |
| `POST` | `/api/interview/add-rule` | Adds custom business definitions and SQL filter hints |
| `POST` | `/api/measures/create` | Compiles, verifies, and persists bidirectional DAX $\leftrightarrow$ SQL measures |
| `POST` | `/api/measures/calculated-column` | Compiles and validates calculated column expressions |
| `POST` | `/api/chat/query` | Executes governed conversational BI query with self-healing retry loop |
| `POST` | `/api/chat/chart` | Generates Plotly visualization for arbitrary tabular rows |
| `POST` | `/api/chat/teach` | Learns human corrective feedback into the business rule RAG store |
| `POST` | `/api/chat/clear` | Clears chat history for the current session workspace |
| `POST` | `/api/root-cause` | Period-over-period variance decomposition and waterfall generation |
| `POST` | `/api/dashboard/pin` | Pins an analytics widget to the executive dashboard |
| `DELETE` | `/api/dashboard/pin/{id}` | Removes a pinned widget |
| `POST` | `/api/dashboard/simulate` | Interactive what-if revenue & cost shock simulation |
| `POST` | `/api/sql/execute` | Executes read-only SQL/DAX query via the PrivacyFirewall AST sandbox |
| `GET` | `/api/export/report` | Generates a printable Executive BI Report (`.html` / Print-to-PDF) |
| `GET` | `/api/export/okf` | Downloads the governed `okf.yaml` semantic layer |
| `GET` | `/api/export/dax` | Downloads the Power BI `.dax` measures script |

---

## 🧪 Automated Testing & CI

Run the complete 82-test automated test suite:

```bash
# Run all tests
python -m pytest tests/ -q

# Run with coverage report
python -m pytest --cov=src --cov-report=term-missing
```
