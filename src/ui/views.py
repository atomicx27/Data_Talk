"""Streamlit Workbench Views for Data_Talk.

Implements the 6 core workbench views across the 2-Group Workflow Navigation:
Group 1 — Semantic Model Studio:
  1. Connect & Profile (1-Click Enterprise Demo, Excel/CSV/Parquet, SQL DBs, Power BI .pbix/Desktop/Cloud, PII & Privacy Blocklist)
  2. Joins & ER Map (Interactive 3-Signal Join Approval Table + 1-Hop Focal Graphviz ER Diagram)
  3. AI Business Interview (Prioritized Ambiguity & Status Code Q&A + Readiness Score + Skip to Chat)
  4. Semantic Measure & DAX Studio (Create/Verify DAX, SQL, & NL Measures + Calculated Columns + Starter Templates)

Group 2 — Analytics Workbench:
  5. Talk to Data Chat (Conversational BI, Interactive Plotly Switcher, Lineage & Explain Drawer, 1-Click Teach AI, Root-Cause 'Why' Analyzer)
  6. Live Dashboard & What-If Canvas (Pinned Widgets, What-If Sliders, Export OKF/.dax/CSV, and Outbound LLM Payload Inspector)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
import yaml

from src.agents.interviewer_agent import InterviewerAgent
from src.agents.llm_router import LLMRouter
from src.agents.measure_compiler import MeasureCompiler
from src.agents.query_agent import QueryAgent, QueryResponse
from src.agents.root_cause_agent import RootCauseAgent
from src.connectors.base import BaseConnector
from src.connectors.duckdb_connector import DuckDBConnector
from src.connectors.powerbi_connector import PowerBIConnector
from src.connectors.profiler import StatisticalProfiler
from src.connectors.sql_connector import SQLConnector
from src.knowledge.join_inference import JoinInferenceEngine
from src.knowledge.okf_models import (
    BusinessRule,
    JoinCardinality,
    JoinStatus,
    OKFWorkspace,
    PinnedWidget,
    RelationshipEdge,
)
from src.knowledge.okf_store import OKFStore
from src.knowledge.rag_store import RAGStore
from src.security.privacy_firewall import CredentialVault
from src.viz.chart_recommender import ChartRecommender


def run_full_onboarding_pipeline(
    workspace: OKFWorkspace,
    connector: BaseConnector,
    okf_store: OKFStore,
    rag_store: RAGStore,
    llm_router: LLMRouter,
    blocked_tables: set[str] | None = None,
    blocked_columns: set[str] | None = None,
) -> None:
    """Execute the automated Schema Extraction -> Push-Down Profiling -> 3-Signal Join Inference -> Interview Setup pipeline."""
    with st.status("Running Data_Talk Onboarding Pipeline...", expanded=True) as status:
        st.write("1/4 Extracting tables, columns, primary keys & explicit relationships...")
        raw_tables, explicit_rels, imported_measures = connector.extract_schema(
            blocked_tables=blocked_tables,
            blocked_columns=blocked_columns,
        )

        st.write(f"2/4 Running push-down statistical profiler & PII scanner across {len(raw_tables)} table(s)...")
        profiled_tables = StatisticalProfiler.profile_tables(
            connector,
            raw_tables,
            blocked_tables=blocked_tables,
            blocked_columns=blocked_columns,
        )
        workspace.tables = profiled_tables

        st.write("3/4 Running 3-Signal Join Inference Engine (FKs + Lexical + Value Overlap)...")
        workspace.relationships = JoinInferenceEngine.infer_relationships(
            profiled_tables,
            existing_relationships=explicit_rels,
            connector=connector,
        )

        if imported_measures:
            existing_ids = {m.id for m in workspace.measures}
            for im in imported_measures:
                if im.id not in existing_ids:
                    workspace.measures.append(im)

        st.write("4/4 Indexing OKF into Hybrid RAG & generating prioritized Business Interview questions...")
        okf_store.classify_table_roles(workspace)
        InterviewerAgent.sync_schema_to_rag(workspace, rag_store)
        InterviewerAgent.generate_prioritized_questions(workspace, llm_router=llm_router)
        okf_store.save(workspace)
        status.update(label="Onboarding Pipeline Complete! Schema, Joins, & Interview Questions Ready.", state="complete")


# ===========================================================================
# VIEW 1: Connect & Profile
# ===========================================================================
def render_connect_view(
    workspace: OKFWorkspace,
    connector: BaseConnector,
    okf_store: OKFStore,
    rag_store: RAGStore,
    llm_router: LLMRouter,
    vault: CredentialVault,
) -> None:
    st.subheader("1. Connect Data Source & Statistical Profiler")
    st.caption(
        "Connect Excel/CSV/Parquet, SQL Databases, or Power BI (.pbix, Live Desktop, or Cloud REST API). "
        "Data_Talk profiles cardinality, detects PII automatically, and builds your OKF Semantic Graph."
    )

    # Top Quick-Launch Banner for 1-Click 4-Table Enterprise Demo
    with st.container(border=True):
        c1, c2 = st.columns([3, 1])
        with c1:
            st.markdown("#### 🚀 Instant 30-Second Test: Load 1-Click Multi-Table Enterprise Demo")
            st.markdown(
                "Generates a realistic 4-table e-commerce warehouse (`customers`, `products`, `orders`, `returns` — 1,970 rows) "
                "with cryptic columns (`ord_stat`, `rev_gross_lcy`, `cust_tier`, `ret_reason_cd`) and PII (`contact_email`) "
                "so you can immediately test **Join Approval**, **AI Business Interview**, **DAX Measure Studio**, and **Talk-to-Data Chat**."
            )
        with c2:
            if st.button("⚡ Load 4-Table Demo Dataset", type="primary", use_container_width=True):
                if isinstance(connector, DuckDBConnector):
                    connector.load_builtin_demo_dataset()
                    workspace.source_type = "duckdb"
                    workspace.dialect = "duckdb"
                    run_full_onboarding_pipeline(workspace, connector, okf_store, rag_store, llm_router)
                    # Pre-load starter DAX measures so user has examples ready
                    for tmpl in MeasureCompiler.STARTER_MEASURE_TEMPLATES[:2]:
                        MeasureCompiler.create_and_verify_measure(
                            workspace=workspace,
                            rag_store=rag_store,
                            connector=connector,
                            name=tmpl["name"],
                            table_name=tmpl["table_name"],
                            user_formula=tmpl["dax_expression"],
                            input_type="dax",
                            description=tmpl["description"],
                            llm_router=llm_router,
                        )
                    okf_store.save(workspace)
                    st.rerun()

    src_tab1, src_tab2, src_tab3 = st.tabs(
        ["📂 Excel / CSV / Parquet", "🗄️ SQL Databases (Push-Down)", "📊 Power BI (.pbix / Desktop / Cloud)"]
    )

    with src_tab1:
        uploaded_files = st.file_uploader(
            "Upload one or more CSV, Excel (.xlsx/.xls), or Parquet files (multi-sheet Excel automatically splits into tables)",
            type=["csv", "xlsx", "xls", "parquet"],
            accept_multiple_files=True,
        )
        if uploaded_files and st.button("Ingest & Profile Uploaded Files", type="primary"):
            upload_dir = okf_store.workspace_dir(workspace.workspace_id) / "uploads"
            upload_dir.mkdir(parents=True, exist_ok=True)
            duck_conn = DuckDBConnector(okf_store.workspace_dir(workspace.workspace_id))
            for uf in uploaded_files:
                dest = upload_dir / uf.name
                dest.write_bytes(uf.getvalue())
                duck_conn.ingest_file(dest, original_filename=uf.name)
            st.session_state["active_connector"] = duck_conn
            workspace.source_type = "duckdb"
            workspace.dialect = "duckdb"
            run_full_onboarding_pipeline(workspace, duck_conn, okf_store, rag_store, llm_router)
            st.rerun()

    with src_tab2:
        st.markdown("Connect to **PostgreSQL**, **MySQL**, **SQL Server**, **SQLite**, or **Snowflake** (Read-Only Push-Down Mode).")
        col_d, col_ref = st.columns([1, 2])
        with col_d:
            sql_dialect = st.selectbox("SQL Dialect", ["postgres", "mysql", "tsql", "sqlite", "snowflake"])
        with col_ref:
            cred_name = st.text_input("Symbolic Credential Name (stored in encrypted `.secrets/vault.enc`, never in `okf.yaml`)", value="prod_sql_db")
        conn_url = st.text_input(
            "SQLAlchemy Connection URL",
            placeholder="postgresql+psycopg2://readonly_user:password@localhost:5432/analytics_db or sqlite:///path/to/db.sqlite",
            type="password",
        )
        if st.button("Connect & Profile SQL Database"):
            if not conn_url.strip():
                st.error("Please provide a valid SQLAlchemy Connection URL.")
            else:
                try:
                    vault.set_secret(cred_name, conn_url.strip())
                    sql_conn = SQLConnector(conn_url.strip(), dialect_hint=sql_dialect)
                    st.session_state["active_connector"] = sql_conn
                    workspace.source_type = "sql"
                    workspace.dialect = sql_conn.dialect
                    workspace.credential_ref = cred_name
                    run_full_onboarding_pipeline(workspace, sql_conn, okf_store, rag_store, llm_router)
                    st.rerun()
                except Exception as exc:
                    st.error(f"SQL Connection Failed: {exc}")

    with src_tab3:
        pbi_mode = st.radio(
            "Select Power BI Connection Mode",
            [
                "Offline .pbix / .pbit File Upload (Extracts tables, relationships, & DAX measures into DuckDB)",
                "Live Power BI Desktop on Windows (Auto-discovers local Analysis Services localhost:<port>)",
                "Power BI Cloud Service REST API (Executes DAX via datasets/{id}/executeQueries)",
            ],
        )
        if pbi_mode.startswith("Offline"):
            pbix_file = st.file_uploader("Upload a Power BI `.pbix` or `.pbit` file", type=["pbix", "pbit"])
            if pbix_file and st.button("Extract & Profile .pbix Model"):
                upload_dir = okf_store.workspace_dir(workspace.workspace_id) / "uploads"
                upload_dir.mkdir(parents=True, exist_ok=True)
                dest = upload_dir / pbix_file.name
                dest.write_bytes(pbix_file.getvalue())
                try:
                    pbi_conn = PowerBIConnector(
                        mode="powerbi_pbix",
                        workspace_dir=okf_store.workspace_dir(workspace.workspace_id),
                        pbix_path=dest,
                    )
                    st.session_state["active_connector"] = pbi_conn
                    workspace.source_type = "powerbi_pbix"
                    workspace.dialect = "duckdb"
                    run_full_onboarding_pipeline(workspace, pbi_conn, okf_store, rag_store, llm_router)
                    st.rerun()
                except Exception as exc:
                    st.error(f"Failed to unpack .pbix file: {exc}")

        elif pbi_mode.startswith("Live"):
            instances = PowerBIConnector.discover_local_desktop_instances()
            if instances:
                st.success(f"Detected {len(instances)} running local Power BI Desktop Analysis Services instance(s)!")
                opts = {f"{inst['workspace_folder']} (localhost:{inst['port']})": inst["port"] for inst in instances}
                chosen_label = st.selectbox("Select Running Power BI Desktop Instance", list(opts.keys()))
                port_val = opts[chosen_label]
            else:
                st.info("No running local Power BI Desktop instance auto-detected. Open a `.pbix` in Power BI Desktop or enter port manually:")
                port_val = int(st.number_input("Localhost Analysis Services Port", min_value=1024, max_value=65535, value=55000))

            if st.button("Connect to Live Power BI Desktop"):
                try:
                    pbi_conn = PowerBIConnector(mode="powerbi_desktop", localhost_port=int(port_val))
                    st.session_state["active_connector"] = pbi_conn
                    workspace.source_type = "powerbi_desktop"
                    workspace.dialect = "dax"
                    run_full_onboarding_pipeline(workspace, pbi_conn, okf_store, rag_store, llm_router)
                    st.rerun()
                except Exception as exc:
                    st.error(str(exc))

        else:
            ds_id = st.text_input("Power BI Dataset / Semantic Model GUID")
            token_in = st.text_input("Azure AD Bearer Token (or configure Service Principal in `.env`)", type="password")
            if st.button("Connect to Power BI Cloud Dataset"):
                try:
                    vault.set_secret("pbi_cloud_token", token_in.strip())
                    pbi_conn = PowerBIConnector(
                        mode="powerbi_cloud",
                        dataset_id=ds_id.strip(),
                        access_token=token_in.strip(),
                    )
                    st.session_state["active_connector"] = pbi_conn
                    workspace.source_type = "powerbi_cloud"
                    workspace.dialect = "dax"
                    workspace.credential_ref = "pbi_cloud_token"
                    run_full_onboarding_pipeline(workspace, pbi_conn, okf_store, rag_store, llm_router)
                    st.rerun()
                except Exception as exc:
                    st.error(f"Power BI Cloud Connection Error: {exc}")

    # Display Profiled Tables, PII Redaction Badges & Privacy Blocklist
    if workspace.tables:
        st.divider()
        st.markdown("### 🔍 Discovered Schema, Statistical Profile & Privacy Blocklist")
        st.caption(
            "Columns flagged with **🛡️ PII Redacted** automatically have their raw values blocked from `okf.yaml` and LLM prompts. "
            "You can also check **Block from AI** on any table or column."
        )

        for tbl in workspace.tables:
            role_color = "🟦 FACT" if tbl.table_role.value == "FACT" else "🟩 DIMENSION"
            with st.expander(
                f"Table: `{tbl.name}` ({role_color} • {tbl.row_count:,} rows • {len(tbl.columns)} columns)",
                expanded=True,
            ):
                if tbl.quality_alerts:
                    for alert in tbl.quality_alerts:
                        st.warning(f"⚠️ Anomaly Watchdog: {alert}")

                col_rows = []
                for c in tbl.columns:
                    col_rows.append(
                        {
                            "Block from AI": c.is_blocked,
                            "Column": c.name,
                            "Type": c.data_type,
                            "Key": "🔑 PK" if c.is_primary_key else ("🔗 FK" if c.is_foreign_key else ""),
                            "PII Shield": "🛡️ Redacted" if c.is_pii else "✓ Safe",
                            "Distinct": c.distinct_count,
                            "Null %": f"{c.null_ratio*100:.1f}%",
                            "Min": c.min_value or "—",
                            "Max": c.max_value or "—",
                            "Sample Values": ", ".join(c.sample_values[:5]) if c.sample_values else "—",
                            "Business Meaning": c.description or "—",
                        }
                    )
                edited_df = st.data_editor(
                    pd.DataFrame(col_rows),
                    key=f"prof_grid_{tbl.name}",
                    use_container_width=True,
                    hide_index=True,
                    disabled=["Column", "Type", "Key", "PII Shield", "Distinct", "Null %", "Min", "Max", "Sample Values"],
                )
                # Persist any blocklist or description edits
                changed = False
                for idx, r in edited_df.iterrows():
                    col_obj = tbl.columns[int(idx)]
                    if bool(r["Block from AI"]) != col_obj.is_blocked:
                        col_obj.is_blocked = bool(r["Block from AI"])
                        changed = True
                if changed:
                    okf_store.save(workspace)
                    st.toast(f"Updated privacy blocklist for `{tbl.name}`.")


# ===========================================================================
# VIEW 2: Join Editor & ER Graph
# ===========================================================================
def render_joins_view(
    workspace: OKFWorkspace,
    okf_store: OKFStore,
    rag_store: RAGStore,
) -> None:
    st.subheader("2. Interactive Join Verification & 1-Hop Focal ER Graph")
    st.caption(
        "Review how tables connect to one another. Data_Talk combines explicit Foreign Keys, column-name similarity, "
        "and push-down sample overlap to propose joins. Approve or edit them here to lock them into the OKF NetworkX Graph."
    )

    if not workspace.active_tables():
        st.info("No tables connected yet. Go to **1. Connect & Profile** and click **Load 4-Table Demo Dataset**.")
        return

    c_btn1, c_btn2, _ = st.columns([1, 1, 2])
    with c_btn1:
        if st.button("✅ Approve All Candidate Joins", type="primary", use_container_width=True):
            for rel in workspace.relationships:
                rel.status = JoinStatus.APPROVED
            okf_store.save(workspace)
            st.toast("All relationships approved and locked into OKF Graph!")
            st.rerun()

    if workspace.relationships:
        rel_rows = []
        for rel in workspace.relationships:
            rel_rows.append(
                {
                    "ID": rel.id,
                    "Source Table": rel.source_table,
                    "Source Column": rel.source_column,
                    "Join Type": rel.join_type,
                    "Target Table": rel.target_table,
                    "Target Column": rel.target_column,
                    "Cardinality": rel.cardinality.value,
                    "Confidence": f"{int(rel.confidence * 100)}%",
                    "Overlap Ratio": f"{rel.overlap_ratio * 100:.1f}%",
                    "Status": rel.status.value,
                }
            )
        edited_rels = st.data_editor(
            pd.DataFrame(rel_rows),
            key="join_editor_grid",
            use_container_width=True,
            hide_index=True,
            column_config={
                "Status": st.column_config.SelectboxColumn(
                    "Approval Status",
                    options=["approved", "pending", "rejected"],
                    required=True,
                ),
                "Join Type": st.column_config.SelectboxColumn(
                    "Join Type",
                    options=["LEFT", "INNER", "FULL"],
                    required=True,
                ),
                "Cardinality": st.column_config.SelectboxColumn(
                    "Cardinality",
                    options=["N:1", "1:N", "1:1", "N:M"],
                    required=True,
                ),
            },
            disabled=["ID", "Confidence", "Overlap Ratio"],
        )

        if st.button("💾 Save Join Changes to OKF"):
            for idx, row in edited_rels.iterrows():
                rel_obj = workspace.relationships[int(idx)]
                rel_obj.source_table = str(row["Source Table"])
                rel_obj.source_column = str(row["Source Column"])
                rel_obj.join_type = str(row["Join Type"])
                rel_obj.target_table = str(row["Target Table"])
                rel_obj.target_column = str(row["Target Column"])
                rel_obj.cardinality = JoinCardinality(str(row["Cardinality"]))
                rel_obj.status = JoinStatus(str(row["Status"]))
            okf_store.save(workspace)
            st.success("Join configuration saved to `okf.yaml`!")

    # Add Custom Join Form
    with st.expander("➕ Add a Custom Table Relationship"):
        table_names = [t.name for t in workspace.active_tables()]
        ca, cb, cc, cd, ce = st.columns(5)
        with ca:
            s_tbl = st.selectbox("Source Table", table_names, key="new_rel_st")
        with cb:
            s_cols = [c.name for c in (workspace.get_table(s_tbl).columns if workspace.get_table(s_tbl) else [])]
            s_col = st.selectbox("Source Column", s_cols, key="new_rel_sc")
        with cc:
            t_tbl = st.selectbox("Target Table", table_names, index=min(1, len(table_names) - 1), key="new_rel_tt")
        with cd:
            t_cols = [c.name for c in (workspace.get_table(t_tbl).columns if workspace.get_table(t_tbl) else [])]
            t_col = st.selectbox("Target Column", t_cols, key="new_rel_tc")
        with ce:
            card = st.selectbox("Cardinality", ["N:1", "1:1", "1:N"], key="new_rel_card")
        if st.button("Add & Approve Relationship"):
            new_id = f"user_rel_{s_tbl}_{s_col}_to_{t_tbl}_{t_col}"
            workspace.relationships.append(
                RelationshipEdge(
                    id=new_id,
                    source_table=s_tbl,
                    source_column=s_col,
                    target_table=t_tbl,
                    target_column=t_col,
                    cardinality=JoinCardinality(card),
                    confidence=1.0,
                    discovery_method="user_defined",
                    status=JoinStatus.APPROVED,
                    overlap_ratio=1.0,
                )
            )
            okf_store.save(workspace)
            st.rerun()

    st.divider()
    st.markdown("### 🕸️ Entity-Relationship (ER) Semantic Graph")
    all_t_names = [t.name for t in workspace.active_tables()]
    focal = st.multiselect(
        "Filter Focal Tables (shows selected tables ± 1-hop neighbors to prevent clutter on 100+ table schemas)",
        options=all_t_names,
        default=all_t_names[:6],
    )
    dot_source = okf_store.generate_er_graphviz(workspace, focal_tables=focal or None)
    st.graphviz_chart(dot_source, use_container_width=True)


# ===========================================================================
# VIEW 3: AI Business Interview
# ===========================================================================
def render_interview_view(
    workspace: OKFWorkspace,
    okf_store: OKFStore,
    rag_store: RAGStore,
    llm_router: LLMRouter,
) -> None:
    st.subheader("3. Smart AI Business Knowledge Interview")
    st.caption(
        "Data_Talk prioritizes the highest-impact business ambiguities in your schema (status codes, revenue definitions, "
        "cryptic column names, and fiscal calendars). Every answer is saved into both `okf.yaml` and the Vector RAG store."
    )

    if not workspace.active_tables():
        st.info("Connect a dataset in **1. Connect & Profile** first.")
        return

    if not workspace.interview_questions:
        InterviewerAgent.generate_prioritized_questions(workspace, llm_router=llm_router)
        okf_store.save(workspace)

    top_c1, top_c2 = st.columns([3, 1])
    with top_c1:
        answered_cnt = sum(1 for q in workspace.interview_questions if q.answered)
        total_cnt = len(workspace.interview_questions)
        st.progress(
            answered_cnt / max(1, total_cnt),
            text=f"Business Interview Progress: {answered_cnt} / {total_cnt} high-impact questions answered",
        )
    with top_c2:
        if st.button("⏭️ Skip / Go Straight to Chat", use_container_width=True):
            st.session_state["nav_step"] = "5. Talk to Data Chat"
            st.rerun()

    for q in workspace.interview_questions:
        badge = "✅ Saved to OKF & RAG" if q.answered else f"🔥 Priority P{q.priority} ({q.category})"
        with st.container(border=True):
            st.markdown(f"**{q.question_text}**  \n`{badge}` • _{q.context_hint}_")

            if q.suggested_options:
                st.caption("Quick-select a recommended business rule or type your own below:")
                btn_cols = st.columns(len(q.suggested_options))
                for idx, opt in enumerate(q.suggested_options):
                    if btn_cols[idx].button(
                        f"Use: {opt[:65]}{'...' if len(opt) > 65 else ''}",
                        key=f"sug_{q.id}_{idx}",
                    ):
                        InterviewerAgent.record_answer(workspace, rag_store, q.id, opt)
                        okf_store.save(workspace)
                        st.rerun()

            ans_col, save_col = st.columns([4, 1])
            with ans_col:
                user_ans = st.text_input(
                    "Your Business Definition / Rule",
                    value=q.answer,
                    placeholder="Type business explanation or click a recommended rule above...",
                    key=f"ans_input_{q.id}",
                    label_visibility="collapsed",
                )
            with save_col:
                if st.button("Save Rule", key=f"save_btn_{q.id}", use_container_width=True):
                    InterviewerAgent.record_answer(workspace, rag_store, q.id, user_ans)
                    okf_store.save(workspace)
                    st.toast("Saved business rule to OKF & RAG!")
                    st.rerun()

    # Manual Business Glossary / Rule Entry
    with st.expander("➕ Add Custom Business Glossary Term or Domain Rule"):
        r_term = st.text_input("Business Term / Metric / Acronym (e.g., 'NRR', 'Active Enterprise Account', 'Churned')")
        r_def = st.text_area("Business Definition & Calculation Rule")
        r_sql = st.text_input("Optional SQL Filter Hint (e.g., `customers.cust_tier = 'ENT' AND orders.ord_stat = 'C'`)")
        if st.button("Add Business Rule to OKF & RAG"):
            if r_term.strip() and r_def.strip():
                rid = f"manual_rule_{len(workspace.business_rules) + 1}"
                rule = BusinessRule(
                    id=rid,
                    term=r_term.strip(),
                    definition=r_def.strip(),
                    sql_filter_hint=r_sql.strip(),
                    source="manual",
                )
                workspace.business_rules.append(rule)
                rag_store.upsert(
                    "business_rules",
                    rid,
                    f"Term: {r_term}\nDefinition: {r_def}\nSQL Filter: {r_sql}",
                    metadata={"term": r_term, "sql_filter_hint": r_sql},
                )
                okf_store.save(workspace)
                st.success(f"Added `{r_term}` to OKF & RAG memory!")
                st.rerun()


# ===========================================================================
# VIEW 4: Semantic Measure & DAX Studio
# ===========================================================================
def render_measure_studio_view(
    workspace: OKFWorkspace,
    connector: BaseConnector,
    okf_store: OKFStore,
    rag_store: RAGStore,
    llm_router: LLMRouter,
) -> None:
    st.subheader("4. Unified Semantic Measure & Calculated Column Studio (DAX ↔ SQL)")
    st.caption(
        "Define reusable Measures and Calculated Columns using **Power BI DAX**, **SQL**, or **Plain English**. "
        "Data_Talk compiles both representations, dry-run verifies them on live data, and indexes your DAX into the RAG Few-Shot library."
    )

    if not workspace.active_tables():
        st.info("Connect a dataset in **1. Connect & Profile** first.")
        return

    # 1-Click Starter Templates
    with st.expander("⚡ 1-Click Power BI DAX Starter Measure Templates", expanded=not bool(workspace.measures)):
        t_cols = st.columns(len(MeasureCompiler.STARTER_MEASURE_TEMPLATES))
        for idx, tmpl in enumerate(MeasureCompiler.STARTER_MEASURE_TEMPLATES):
            with t_cols[idx]:
                st.markdown(f"**{tmpl['name']}**")
                st.code(tmpl["dax_expression"], language="dax")
                if st.button(f"Add '{tmpl['name']}'", key=f"tmpl_meas_{idx}", use_container_width=True):
                    MeasureCompiler.create_and_verify_measure(
                        workspace=workspace,
                        rag_store=rag_store,
                        connector=connector,
                        name=tmpl["name"],
                        table_name=tmpl["table_name"],
                        user_formula=tmpl["dax_expression"],
                        input_type="dax",
                        description=tmpl["description"],
                        llm_router=llm_router,
                    )
                    okf_store.save(workspace)
                    st.rerun()

    tab_meas, tab_calc = st.tabs(["📐 Create Semantic Measure (Aggregation)", "➕ Create Calculated Column (Row-Level)"])

    table_names = [t.name for t in workspace.active_tables()]

    with tab_meas:
        mc1, mc2, mc3 = st.columns([2, 1, 1])
        with mc1:
            m_name = st.text_input("Measure Name", placeholder="e.g., Net Margin % or Completed Enterprise Revenue")
        with mc2:
            m_table = st.selectbox("Home Table", table_names, key="meas_tbl")
        with mc3:
            m_type = st.selectbox("Formula Language", ["dax", "sql", "natural_language"])

        m_formula = st.text_area(
            "Formula / Prompt",
            placeholder=(
                "DAX example: DIVIDE(SUM(orders[profit_usd]), SUM(orders[rev_net_usd]), 0)\n"
                "or CALCULATE(SUM(orders[rev_net_usd]), orders[ord_stat] = \"C\")\n"
                "or Natural Language: Total profit divided by net revenue excluding cancelled orders"
            ),
            height=90,
        )
        m_desc = st.text_input("Business Description (when should the AI use this measure?)")

        if st.button("🔨 Compile, Dry-Run Verify & Save Measure to OKF", type="primary"):
            if not (m_name.strip() and m_formula.strip()):
                st.error("Please provide both a Measure Name and a Formula.")
            else:
                meas = MeasureCompiler.create_and_verify_measure(
                    workspace=workspace,
                    rag_store=rag_store,
                    connector=connector,
                    name=m_name.strip(),
                    table_name=m_table,
                    user_formula=m_formula.strip(),
                    input_type=m_type,
                    description=m_desc.strip(),
                    llm_router=llm_router,
                )
                okf_store.save(workspace)
                if meas.verified:
                    st.success(f"Verified `{meas.name}`! Live Sample Value = **{meas.sample_value}**")
                else:
                    st.warning(f"Saved `{meas.name}` to OKF (dry-run preview skipped or requires multi-join context).")

    with tab_calc:
        cc1, cc2, cc3 = st.columns([2, 1, 1])
        with cc1:
            c_name = st.text_input("Calculated Column Name", placeholder="e.g., deal_size_bucket")
        with cc2:
            c_table = st.selectbox("Attach to Table", table_names, key="calc_tbl")
        with cc3:
            c_type = st.selectbox("Expression Type", ["dax", "sql"], key="calc_type")

        c_formula = st.text_input(
            "Row-Level Formula",
            placeholder='IF(orders[rev_net_usd] >= 5000, "Large Deal", "Standard Deal")',
        )
        if st.button("🔨 Compile & Verify Calculated Column"):
            if c_name.strip() and c_formula.strip():
                cc_obj = MeasureCompiler.create_and_verify_calculated_column(
                    workspace=workspace,
                    rag_store=rag_store,
                    connector=connector,
                    table_name=c_table,
                    column_name=c_name.strip(),
                    user_formula=c_formula.strip(),
                    input_type=c_type,
                    llm_router=llm_router,
                )
                okf_store.save(workspace)
                st.success(f"Saved Calculated Column `{c_table}[{c_name}]`! Preview: {cc_obj.sample_preview}")

    # List Governed Measures & Calculated Columns
    if workspace.measures:
        st.divider()
        st.markdown("### 📚 Governed Semantic Measures in OKF & RAG Few-Shot Library")
        for m in workspace.measures:
            v_badge = "✅ Verified on Data" if m.verified else "📝 Stored"
            val_str = f" • **Live Value:** `{m.sample_value:,.4f}`" if isinstance(m.sample_value, (int, float)) else ""
            with st.container(border=True):
                st.markdown(f"#### `{m.table_name}[{m.name}]`  ({v_badge}{val_str})")
                st.caption(m.description)
                d_col, s_col = st.columns(2)
                with d_col:
                    st.markdown("**Power BI DAX (`DEFINE MEASURE`):**")
                    st.code(f"MEASURE '{m.table_name}'[{m.name}] = {m.dax_expression}", language="dax")
                with s_col:
                    st.markdown("**Compiled SQL / DuckDB Expression:**")
                    st.code(m.sql_expression, language="sql")


# ===========================================================================
# VIEW 5: Talk to Data Chat
# ===========================================================================
def render_chat_view(
    workspace: OKFWorkspace,
    connector: BaseConnector,
    okf_store: OKFStore,
    rag_store: RAGStore,
    llm_router: LLMRouter,
) -> None:
    st.subheader("5. Talk to Data — Conversational BI, Auto-Charts & Lineage")

    if not workspace.active_tables():
        st.info("No dataset loaded yet. Go to **1. Connect & Profile** and click **⚡ Load 4-Table Demo Dataset**.")
        return

    # Starter Prompts Bar
    starter_prompts = [
        "Show net revenue, profit, and gross margin % by product category",
        "Show monthly net revenue trend over time",
        "Compare net revenue, refunds, and return rate % by product category",
        "Show revenue and order count by customer tier",
        "Show breakdown of returns by return reason",
    ]
    st.caption("💡 Click a starter question or ask anything in natural language:")
    s_cols = st.columns(len(starter_prompts))
    triggered_prompt: str | None = None
    for idx, sp in enumerate(starter_prompts):
        if s_cols[idx].button(sp, key=f"starter_btn_{idx}", use_container_width=True):
            triggered_prompt = sp

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    user_input = st.chat_input("Ask a question about your data (e.g., 'What is net revenue and margin % by region?')...")
    active_prompt = triggered_prompt or user_input

    if active_prompt:
        what_if_overrides = st.session_state.get("what_if_overrides", {})
        resp, df = QueryAgent.answer_question(
            question=active_prompt,
            workspace=workspace,
            okf_store=okf_store,
            rag_store=rag_store,
            connector=connector,
            llm_router=llm_router,
            what_if_overrides=what_if_overrides,
        )
        resp.recommended_chart_type = ChartRecommender.recommend_chart_type(df, active_prompt)
        st.session_state["chat_history"].append({"response": resp, "df": df})

    # Render Chat History
    for idx, item in enumerate(st.session_state["chat_history"]):
        resp: QueryResponse = item["response"]
        df: pd.DataFrame = item["df"]

        with st.chat_message("user"):
            st.markdown(f"**{resp.user_question}**")

        with st.chat_message("assistant"):
            if not resp.success:
                st.error(f"Query Failed after {resp.attempts_used} attempts: `{resp.error_message}`")
                if resp.healing_trace:
                    st.caption("Self-Healing Trace: " + " → ".join(resp.healing_trace))
                st.code(resp.executed_query, language="sql")
                continue

            st.markdown(resp.executive_summary)

            if resp.clarification_suggestion:
                st.info(f"🧠 **Just-in-Time Business Learning:** {resp.clarification_suggestion}")

            # Single-Row KPI Metrics Display
            if len(df) == 1:
                num_cols = list(df.select_dtypes(include=["number"]).columns)
                if num_cols:
                    k_cols = st.columns(min(4, len(num_cols)))
                    for k_idx, col_name in enumerate(num_cols[:4]):
                        val = float(df[col_name].iloc[0] or 0)
                        suffix = "%" if "pct" in col_name.lower() or "margin" in col_name.lower() else ""
                        k_cols[k_idx].metric(label=str(col_name), value=f"{val:,.2f}{suffix}")

            # Multi-Row Interactive Plotly Chart + Chart Switcher + Pin Button
            elif len(df) > 1:
                ctrl1, ctrl2, ctrl3 = st.columns([2, 2, 1])
                with ctrl1:
                    default_idx = (
                        ChartRecommender.SUPPORTED_CHART_TYPES.index(resp.recommended_chart_type)
                        if resp.recommended_chart_type in ChartRecommender.SUPPORTED_CHART_TYPES
                        else 0
                    )
                    chosen_chart = st.selectbox(
                        "Chart Type",
                        ChartRecommender.SUPPORTED_CHART_TYPES,
                        index=default_idx,
                        key=f"chart_sel_{idx}",
                    )
                with ctrl3:
                    if st.button("📌 Pin to Dashboard", key=f"pin_btn_{idx}", use_container_width=True):
                        widget = PinnedWidget(
                            id=f"pin_{len(workspace.pinned_widgets) + 1}_{idx}",
                            title=resp.user_question[:60],
                            user_question=resp.user_question,
                            executed_query=resp.executed_query,
                            query_dialect=resp.query_dialect,
                            chart_type=chosen_chart,
                        )
                        workspace.pinned_widgets.append(widget)
                        okf_store.save(workspace)
                        st.toast("Pinned chart to Live Dashboard!")

                fig = ChartRecommender.build_figure(df, chart_type=chosen_chart, title=resp.user_question)
                if fig is not None:
                    st.plotly_chart(fig, use_container_width=True, key=f"plotly_fig_{idx}")

            with st.expander(f"📋 View Underlying Data Table ({len(df):,} rows)", expanded=False):
                st.dataframe(df, use_container_width=True)

            # "Explain This Number" Lineage & Governance Drawer
            with st.expander("🔍 Explain This Number — SQL/DAX Lineage, Approved Joins & Teach AI"):
                if resp.fan_trap_mitigated:
                    st.success(
                        "🛡️ **Multi-Fact Fan/Chasm Trap Mitigated:** Separate pre-aggregated CTEs were used for each Fact table before joining."
                    )
                if resp.healing_trace:
                    st.info("🩹 **Self-Healing Query Trace:**\n- " + "\n- ".join(resp.healing_trace))

                l_col1, l_col2 = st.columns(2)
                with l_col1:
                    st.markdown(f"**Executed Query (`{resp.query_dialect}`):**")
                    st.code(resp.executed_query, language="sql")
                with l_col2:
                    st.markdown("**Companion Power BI DAX Query (`EVALUATE`):**")
                    st.code(resp.compiled_dax_equivalent, language="dax")

                st.markdown(
                    f"- **Tables Used:** `{', '.join(resp.involved_tables)}`\n"
                    f"- **OKF Join Paths:** `{'; '.join(resp.join_clauses_used) or 'Single-table query'}`\n"
                    f"- **Governed Measures Applied:** `{', '.join(resp.measures_applied) or 'Standard Aggregations'}`"
                )

                # 1-Click Teach / Correct AI inline form
                teach_rule = st.text_input(
                    "🎓 Teach / Correct AI (save a permanent business rule to OKF & RAG from this query):",
                    placeholder="e.g., Exclude ord_stat = 'X' from all revenue calculations",
                    key=f"teach_in_{idx}",
                )
                if st.button("Save Correction to OKF", key=f"teach_btn_{idx}"):
                    if teach_rule.strip():
                        rid = f"chat_rule_{len(workspace.business_rules) + 1}"
                        sql_hint = "orders.ord_stat <> 'X'" if "ord_stat" in teach_rule or "'X'" in teach_rule else ""
                        workspace.business_rules.append(
                            BusinessRule(
                                id=rid,
                                term=f"Correction on: {resp.user_question[:35]}",
                                definition=teach_rule.strip(),
                                sql_filter_hint=sql_hint,
                                source="chat_correction",
                            )
                        )
                        rag_store.upsert("business_rules", rid, teach_rule.strip())
                        okf_store.save(workspace)
                        st.success("Saved rule to OKF & RAG! Future queries will follow this rule.")

    # Root-Cause "Why Did X Change?" Decomposition Analyzer
    if workspace.active_tables():
        st.divider()
        with st.expander("📉 Power BI-Style 'Why Did It Change?' Root-Cause Variance Decomposition"):
            rc1, rc2, rc3 = st.columns([1, 1, 1])
            with rc1:
                rc_metric = st.selectbox("Target Metric", ["rev_net_usd", "profit_usd", "rev_gross_lcy"])
            with rc2:
                rc_dim = st.selectbox("Drill-Down Dimension", ["category", "region", "customer_tier", "sales_channel"])
            with rc3:
                run_rc = st.button("🔍 Explain Period-over-Period Variance", use_container_width=True)

            if run_rc:
                rc_res = RootCauseAgent.analyze_variance(workspace, connector, metric_col=rc_metric, dimension=rc_dim)
                st.markdown(rc_res["summary"])
                rc_df: pd.DataFrame = rc_res["df"]
                if not rc_df.empty:
                    fig_wf = ChartRecommender.build_figure(
                        rc_df,
                        chart_type="waterfall",
                        title=f"Variance Breakdown of {rc_metric} by {rc_dim}",
                        x_col="segment",
                    )
                    if fig_wf:
                        st.plotly_chart(fig_wf, use_container_width=True)
                    st.dataframe(rc_df, use_container_width=True)


# ===========================================================================
# VIEW 6: Live Dashboard, What-If Sliders & Outbound Privacy Inspector
# ===========================================================================
def render_dashboard_view(
    workspace: OKFWorkspace,
    connector: BaseConnector,
    okf_store: OKFStore,
    llm_router: LLMRouter,
) -> None:
    st.subheader("6. Live Pin-to-Dashboard Canvas, What-If Simulation & Security Audit")

    # What-If Scenario Sliders
    with st.container(border=True):
        st.markdown("#### 🎛️ Power BI-Style What-If Scenario Parameters")
        st.caption("Adjust sliders to simulate revenue or cost shocks across Talk-to-Data Chat and Live KPIs:")
        w1, w2, w3 = st.columns(3)
        with w1:
            rev_shock = st.slider("Revenue Multiplier %", min_value=-30, max_value=50, value=0, step=5)
        with w2:
            cost_shock = st.slider("Cost Multiplier %", min_value=-30, max_value=50, value=0, step=5)
        st.session_state["what_if_overrides"] = {
            "Revenue Multiplier %": rev_shock / 100.0,
            "Cost Multiplier %": cost_shock / 100.0,
        }
        with w3:
            if workspace.get_table("orders"):
                rm = 1.0 + (rev_shock / 100.0)
                cm = 1.0 + (cost_shock / 100.0)
                sim_df = connector.execute_query(
                    f"SELECT ROUND(SUM(rev_net_usd * {rm}), 2) AS sim_rev, "
                    f"ROUND(SUM((rev_net_usd * {rm}) - (cogs_usd * {cm})), 2) AS sim_profit "
                    f"FROM orders WHERE ord_stat <> 'X'",
                    max_rows=1,
                )
                if not sim_df.empty:
                    st.metric(
                        "Simulated Net Profit (Excl. Cancelled)",
                        f"${float(sim_df['sim_profit'].iloc[0]):,.2f}",
                        delta=f"Rev {rev_shock:+d}% / Cost {cost_shock:+d}%",
                    )

    # Pinned Widgets Grid
    st.markdown("### 📌 Pinned Dashboard Widgets")
    if not workspace.pinned_widgets:
        st.info("No charts pinned yet. Run any query in **5. Talk to Data Chat** and click **📌 Pin to Dashboard**.")
    else:
        grid_cols = st.columns(2)
        for idx, pw in enumerate(workspace.pinned_widgets):
            with grid_cols[idx % 2]:
                with st.container(border=True):
                    st.markdown(f"**{pw.title}**")
                    try:
                        w_df = connector.execute_query(pw.executed_query, max_rows=500)
                        fig = ChartRecommender.build_figure(w_df, chart_type=pw.chart_type, title=pw.title)
                        if fig:
                            st.plotly_chart(fig, use_container_width=True, key=f"dash_fig_{pw.id}")
                        else:
                            st.dataframe(w_df, use_container_width=True)
                    except Exception as exc:
                        st.error(f"Widget refresh error: {exc}")

    st.divider()
    # Export OKF & DAX Measures Script
    ex1, ex2 = st.columns(2)
    with ex1:
        okf_yaml_str = yaml.safe_dump(workspace.model_dump(mode="json"), sort_keys=False)
        st.download_button(
            "⬇️ Export Governed OKF Semantic Model (`okf.yaml`)",
            data=okf_yaml_str,
            file_name=f"{workspace.workspace_id}_okf.yaml",
            mime="text/yaml",
            use_container_width=True,
        )
    with ex2:
        dax_script = MeasureCompiler.format_dax_define_block(workspace.measures) or "-- No measures defined yet"
        st.download_button(
            "⬇️ Export Power BI DAX Measures Script (`.dax`)",
            data=dax_script,
            file_name=f"{workspace.workspace_id}_measures.dax",
            mime="text/plain",
            use_container_width=True,
        )

    # Outbound LLM Payload Inspector (Zero-Data-Leak Verification)
    st.divider()
    with st.expander("🔒 Outbound LLM Payload Inspector & Privacy Firewall Audit Log", expanded=False):
        st.caption(
            "Every prompt sent to LiteLLM is scrubbed by `PrivacyFirewall.inspect_and_scrub_outbound_prompt()` "
            "and recorded here so you can verify that zero raw database rows or credentials ever leave your machine."
        )
        if not llm_router.firewall.audit_log:
            st.write("No outbound LLM API calls recorded in this session yet (Deterministic Local Compiler active).")
        else:
            st.json(llm_router.firewall.audit_log[-10:])
