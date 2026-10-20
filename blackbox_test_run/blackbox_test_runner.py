"""Comprehensive Black-Box Test Harness for Data_Talk Enterprise BI.

Features:
- Full Black-Box interaction via FastAPI REST API endpoints.
- Checkpoint / Resume architecture: tracks progress in `checkpoint.json`.
- Structured interval logging: logs every request, response, error, and assertion.
- Dumps full JSON payloads into `blackbox_test_run/responses/`.
- Detailed error tracking and diagnosis.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

# Ensure workspace root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import server
from src.api import create_app
from src.api.runtime import AppRuntime
from src.security.auth import AuthConfig, Authenticator

LOG_DIR = BASE_DIR / "blackbox_test_run"
LOG_DIR.mkdir(parents=True, exist_ok=True)
RESPONSES_DIR = LOG_DIR / "responses"
RESPONSES_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_FILE = LOG_DIR / "checkpoint.json"
LOG_FILE = LOG_DIR / "blackbox_test_execution.log"

# Configure logging
logger = logging.getLogger("DataTalkBlackBox")
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
file_handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s"))
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
logger.handlers = [file_handler, stream_handler]


class BlackBoxRunner:
    """Orchestrates interval-based black-box test execution with state persistence."""

    PHASES = [
        "phase1_health",
        "phase2_session_setup",
        "phase3_upload_excel",
        "phase4_join_governance",
        "phase5_business_interview",
        "phase6_measures_creation",
        "phase7_conversational_queries",
        "phase8_chart_generation",
        "phase9_what_if_simulation",
        "phase10_pinboard_and_export",
        "phase11_security_firewall",
    ]

    def __init__(self) -> None:
        self.client: TestClient | None = None
        self.state_data: dict[str, Any] = self._load_checkpoint()

    @staticmethod
    def _extract_tables(state: dict[str, Any]) -> list[dict[str, Any]]:
        ws = state.get("workspace", {})
        return ws.get("tables", []) if isinstance(ws, dict) and "tables" in ws else state.get("tables", [])

    @staticmethod
    def _extract_relationships(state: dict[str, Any]) -> list[dict[str, Any]]:
        ws = state.get("workspace", {})
        return ws.get("relationships", []) if isinstance(ws, dict) and "relationships" in ws else state.get("relationships", [])

    @staticmethod
    def _extract_questions(state: dict[str, Any]) -> list[dict[str, Any]]:
        ws = state.get("workspace", {})
        return ws.get("interview_questions", []) if isinstance(ws, dict) and "interview_questions" in ws else state.get("interview_questions", [])

    @staticmethod
    def _extract_readiness_score(state: dict[str, Any]) -> int:
        readiness = state.get("readiness")
        if isinstance(readiness, dict):
            return int(readiness.get("score", 0))
        return int(readiness or 0)

    def _load_checkpoint(self) -> dict[str, Any]:
        if CHECKPOINT_FILE.exists():
            try:
                with CHECKPOINT_FILE.open("r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to read existing checkpoint: {e}. Starting fresh.")
        return {
            "completed_phases": [],
            "current_phase": "phase1_health",
            "phase_metrics": {},
            "errors": [],
            "last_updated": datetime.now().isoformat(),
        }

    def _save_checkpoint(self) -> None:
        self.state_data["last_updated"] = datetime.now().isoformat()
        with CHECKPOINT_FILE.open("w", encoding="utf-8") as f:
            json.dump(self.state_data, f, indent=2)

    def _record_error(self, phase: str, message: str, details: Any = None) -> None:
        err_entry = {
            "timestamp": datetime.now().isoformat(),
            "phase": phase,
            "message": message,
            "details": details,
        }
        self.state_data["errors"].append(err_entry)
        self._save_checkpoint()
        logger.error(f"[{phase}] ERROR RECORDED: {message}")

    def _dump_response(self, name: str, data: Any) -> None:
        dump_path = RESPONSES_DIR / f"{name}.json"
        with dump_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)

    def init_client(self) -> None:
        logger.info("Initializing FastAPI TestClient for black-box testing...")
        server.runtime = AppRuntime(base_dir=BASE_DIR)
        server.auth_config = AuthConfig(api_token=None)
        server.authenticator = Authenticator(server.auth_config)
        app = create_app(runtime=server.runtime, auth_config=server.auth_config)
        self.client = TestClient(app)
        # Ensure session is always pinned to our test workspace
        ws_id = "ws_blackbox_excel"
        self.client.post("/api/settings", json={"workspace_id": ws_id, "privacy_mode": "strict_schema_only"})
        logger.info(f"TestClient initialized and workspace pinned to '{ws_id}'.")

    # -------------------------------------------------------------------------
    # PHASE 1: Health & Readiness
    # -------------------------------------------------------------------------
    def run_phase1_health(self) -> bool:
        phase = "phase1_health"
        logger.info(">>> STARTING PHASE 1: System Health & Provider Readiness Probe")
        t0 = time.time()
        res = self.client.get("/api/health")
        latency_ms = round((time.time() - t0) * 1000, 2)
        
        logger.info(f"GET /api/health -> HTTP {res.status_code} ({latency_ms} ms)")
        if res.status_code != 200:
            self._record_error(phase, f"Health check returned HTTP {res.status_code}", res.text)
            return False
            
        data = res.json()
        self._dump_response("01_health", data)
        
        # Assertions
        assert data.get("status") == "healthy", f"Unexpected status: {data.get('status')}"
        assert data.get("duckdb_version"), "Missing duckdb_version"
        logger.info(f"DuckDB Version: {data.get('duckdb_version')}")
        logger.info(f"Active Workspace: {data.get('active_workspace')}")
        logger.info(f"Providers Ready: {data.get('providers_ready')}")
        
        self.state_data["phase_metrics"][phase] = {
            "status": "PASS",
            "latency_ms": latency_ms,
            "duckdb_version": data.get("duckdb_version"),
            "providers_ready": data.get("providers_ready"),
        }
        return True

    # -------------------------------------------------------------------------
    # PHASE 2: Session & Workspace Setup
    # -------------------------------------------------------------------------
    def run_phase2_session_setup(self) -> bool:
        phase = "phase2_session_setup"
        logger.info(">>> STARTING PHASE 2: Workspace & Session Initialization")
        ws_id = "ws_blackbox_excel"
        
        # 1. Update settings to switch to isolated test workspace
        t0 = time.time()
        req_body = {
            "workspace_id": ws_id,
            "privacy_mode": "strict_schema_only",
        }
        res = self.client.post("/api/settings", json=req_body)
        latency_ms = round((time.time() - t0) * 1000, 2)
        logger.info(f"POST /api/settings -> HTTP {res.status_code} ({latency_ms} ms)")
        if res.status_code != 200:
            self._record_error(phase, f"Settings update returned HTTP {res.status_code}", res.text)
            return False
            
        data = res.json()
        self._dump_response("02_settings", data)
        
        # 2. Verify state reflects active workspace
        res_state = self.client.get("/api/state")
        if res_state.status_code != 200:
            self._record_error(phase, f"Get state returned HTTP {res_state.status_code}", res_state.text)
            return False
        state = res_state.json()
        self._dump_response("02_initial_state", state)
        
        assert state.get("active_workspace_id") == ws_id, f"Active workspace is {state.get('active_workspace_id')}"
        logger.info(f"Workspace set to: {state.get('active_workspace_id')}")
        
        self.state_data["phase_metrics"][phase] = {
            "status": "PASS",
            "workspace_id": ws_id,
        }
        return True

    # -------------------------------------------------------------------------
    # PHASE 3: Multi-Sheet Excel Upload & Ingestion
    # -------------------------------------------------------------------------
    def run_phase3_upload_excel(self) -> bool:
        phase = "phase3_upload_excel"
        logger.info(">>> STARTING PHASE 3: Complex Excel Upload & DuckDB Ingestion")
        excel_path = LOG_DIR / "enterprise_sales_bi.xlsx"
        if not excel_path.exists():
            self._record_error(phase, f"Excel file not found at {excel_path}")
            return False

        file_size_kb = round(excel_path.stat().st_size / 1024, 2)
        logger.info(f"Uploading file: {excel_path.name} ({file_size_kb} KB)...")
        
        t0 = time.time()
        with open(excel_path, "rb") as f:
            files = {"files": (excel_path.name, f, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")}
            res = self.client.post("/api/connect/upload", files=files)
            
        latency_ms = round((time.time() - t0) * 1000, 2)
        logger.info(f"POST /api/connect/upload -> HTTP {res.status_code} ({latency_ms} ms)")
        if res.status_code != 200:
            self._record_error(phase, f"Upload returned HTTP {res.status_code}", res.text)
            return False
            
        state = res.json()
        self._dump_response("03_upload_result_state", state)
        
        # Verify all 6 tables exist and are profiled
        expected_tables = {
            "dim_regions",
            "dim_sales_reps",
            "dim_products",
            "dim_customers",
            "fact_orders",
            "fact_returns",
        }
        loaded_tables = self._extract_tables(state)
        actual_tables = {t["name"] for t in loaded_tables}
        missing = expected_tables - actual_tables
        logger.info(f"Loaded Tables: {actual_tables}")
        if missing:
            self._record_error(phase, f"Missing expected tables: {missing}")
            return False
            
        # Verify row counts and PII redaction
        table_stats = {}
        pii_detected = []
        for t in loaded_tables:
            table_stats[t["name"]] = {
                "row_count": t.get("row_count"),
                "col_count": len(t.get("columns", [])),
                "role": t.get("table_role"),
            }
            for col in t.get("columns", []):
                if col.get("is_pii"):
                    pii_detected.append(f"{t['name']}.{col['name']}")
                    
        logger.info(f"Table Statistics: {table_stats}")
        logger.info(f"PII Redaction Flags Detected: {pii_detected}")
        
        assert "dim_customers.contact_email" in pii_detected, "contact_email was not flagged as PII!"
        assert table_stats["fact_orders"]["row_count"] == 600, f"fact_orders rows: {table_stats['fact_orders']['row_count']}"
        assert table_stats["fact_returns"]["row_count"] == 65, f"fact_returns rows: {table_stats['fact_returns']['row_count']}"
        assert table_stats["dim_customers"]["row_count"] == 60, f"dim_customers rows: {table_stats['dim_customers']['row_count']}"
        assert table_stats["dim_products"]["row_count"] == 15, f"dim_products rows: {table_stats['dim_products']['row_count']}"
        assert table_stats["dim_sales_reps"]["row_count"] == 10, f"dim_sales_reps rows: {table_stats['dim_sales_reps']['row_count']}"
        assert table_stats["dim_regions"]["row_count"] == 4, f"dim_regions rows: {table_stats['dim_regions']['row_count']}"
        
        self.state_data["phase_metrics"][phase] = {
            "status": "PASS",
            "latency_ms": latency_ms,
            "tables_loaded": list(actual_tables),
            "table_stats": table_stats,
            "pii_detected": pii_detected,
        }
        return True

    # -------------------------------------------------------------------------
    # PHASE 4: Join Inference & Relationship Governance
    # -------------------------------------------------------------------------
    def run_phase4_join_governance(self) -> bool:
        phase = "phase4_join_governance"
        logger.info(">>> STARTING PHASE 4: Join Inference & Governance Approval")
        
        # 1. Fetch current relationships
        res = self.client.get("/api/state")
        state = res.json()
        existing_rels = self._extract_relationships(state)
        logger.info(f"Inferred Relationships Count: {len(existing_rels)}")
        for r in existing_rels:
            logger.info(f"  - {r['source_table']}.{r['source_column']} -> {r['target_table']}.{r['target_column']} ({r['cardinality']}) [Status: {r['status']}]")
            
        # 2. Approve all candidate relationships
        t0 = time.time()
        res_approve = self.client.post("/api/joins/approve-all")
        latency_ms = round((time.time() - t0) * 1000, 2)
        logger.info(f"POST /api/joins/approve-all -> HTTP {res_approve.status_code} ({latency_ms} ms)")
        if res_approve.status_code != 200:
            self._record_error(phase, f"Approve-all returned HTTP {res_approve.status_code}", res_approve.text)
            return False
            
        state_after = res_approve.json()
        self._dump_response("04_approved_joins_state", state_after)
        
        current_rels = self._extract_relationships(state_after)
        approved_count = sum(1 for r in current_rels if r.get("status") == "approved")
        logger.info(f"Approved Relationships Count: {approved_count}")
        
        # 3. Add explicit custom joins if not already present
        rels_tuples = {(r["source_table"], r["source_column"], r["target_table"], r["target_column"]) for r in current_rels}
        
        essential_joins = [
            ("fact_orders", "customer_id", "dim_customers", "customer_id"),
            ("fact_orders", "product_id", "dim_products", "product_id"),
            ("fact_orders", "rep_id", "dim_sales_reps", "rep_id"),
            ("dim_customers", "region_id", "dim_regions", "region_id"),
            ("dim_sales_reps", "region_id", "dim_regions", "region_id"),
            ("fact_returns", "order_id", "fact_orders", "order_id"),
        ]
        
        added_count = 0
        for src_t, src_c, tgt_t, tgt_c in essential_joins:
            if (src_t, src_c, tgt_t, tgt_c) not in rels_tuples and (tgt_t, tgt_c, src_t, src_c) not in rels_tuples:
                logger.info(f"Adding essential join: {src_t}.{src_c} -> {tgt_t}.{tgt_c}")
                add_req = {
                    "source_table": src_t,
                    "source_column": src_c,
                    "target_table": tgt_t,
                    "target_column": tgt_c,
                    "join_type": "LEFT",
                    "cardinality": "N:1",
                }
                res_add = self.client.post("/api/joins/add", json=add_req)
                if res_add.status_code == 200:
                    added_count += 1
                else:
                    logger.warning(f"Failed to add custom join {src_t}->{tgt_t}: {res_add.text}")
                    
        logger.info(f"Additional Joins Added: {added_count}")
        
        self.state_data["phase_metrics"][phase] = {
            "status": "PASS",
            "inferred_count": len(existing_rels),
            "approved_count": approved_count,
            "custom_joins_added": added_count,
        }
        return True

    # -------------------------------------------------------------------------
    # PHASE 5: Business Knowledge Interview & Answering Questions
    # -------------------------------------------------------------------------
    def run_phase5_business_interview(self) -> bool:
        phase = "phase5_business_interview"
        logger.info(">>> STARTING PHASE 5: Smart Business Interview & Answering Domain Questions")
        
        res = self.client.get("/api/state")
        state = res.json()
        questions = self._extract_questions(state)
        initial_score = self._extract_readiness_score(state)
        logger.info(f"Initial Readiness Score: {initial_score}%")
        logger.info(f"Generated Domain Questions Count: {len(questions)}")
        
        domain_knowledge = {
            "ord_stat": "Completed: fulfilled orders included in official net revenue; Pending: processing in warehouse; Cancelled: voided orders to be strictly excluded; On-Hold: credit verification review.",
            "ret_reason_cd": "Defective: hardware/software failure; Late Delivery: SLA violation; Wrong Item: packaging discrepancy; Customer Request: voluntary return.",
            "cust_tier": "Enterprise: global Fortune 500 accounts; Mid-Market: growing regional companies; SMB: emerging small enterprises.",
        }
        
        answered_count = 0
        for q in questions:
            qid = q.get("id")
            col = q.get("column_name", "")
            q_text = q.get("question_text", "")
            logger.info(f"Question [{qid}] on '{col}': {q_text[:70]}...")
            
            # Formulate informed business answer
            matched_key = next((k for k in domain_knowledge if k in col.lower() or k in q_text.lower()), None)
            answer_text = domain_knowledge[matched_key] if matched_key else f"Governed analytical field for {col} representing standard commercial dimensions."
            
            t0 = time.time()
            res_ans = self.client.post("/api/interview/answer", json={"question_id": qid, "answer": answer_text})
            lat = round((time.time() - t0) * 1000, 2)
            if res_ans.status_code == 200:
                answered_count += 1
                logger.info(f"  -> Answered successfully ({lat} ms)")
            else:
                logger.warning(f"  -> Answer failed HTTP {res_ans.status_code}: {res_ans.text}")
                
        # Add explicit Business Rule
        rule_req = {
            "term": "Net Revenue Filter Condition",
            "definition": "Official enterprise Net Revenue must filter out Cancelled orders.",
            "sql_filter_hint": "fact_orders.ord_stat <> 'Cancelled'",
        }
        res_rule = self.client.post("/api/interview/add-rule", json=rule_req)
        logger.info(f"POST /api/interview/add-rule -> HTTP {res_rule.status_code}")
        
        # Check readiness score post-interview
        state_after = self.client.get("/api/state").json()
        self._dump_response("05_interview_final_state", state_after)
        final_score = self._extract_readiness_score(state_after)
        logger.info(f"Final Readiness Score: {final_score}% (Gain: +{final_score - initial_score}%)")
        
        self.state_data["phase_metrics"][phase] = {
            "status": "PASS",
            "questions_count": len(questions),
            "answered_count": answered_count,
            "initial_score": initial_score,
            "final_score": final_score,
        }
        return True

    # -------------------------------------------------------------------------
    # PHASE 6: Semantic Measures Studio (Bidirectional DAX <-> SQL)
    # -------------------------------------------------------------------------
    def run_phase6_measures_creation(self) -> bool:
        phase = "phase6_measures_creation"
        logger.info(">>> STARTING PHASE 6: Semantic Measure Studio (DAX <-> SQL Compilation)")
        
        measures_to_create = [
            {
                "name": "Total Gross Revenue",
                "table_name": "fact_orders",
                "user_formula": "SUM(fact_orders[gross_revenue_usd])",
                "input_type": "dax",
                "description": "Total gross invoice amount before customer discounts.",
            },
            {
                "name": "Total Net Revenue",
                "table_name": "fact_orders",
                "user_formula": "SUM(fact_orders[net_revenue_usd])",
                "input_type": "dax",
                "description": "Total recognized revenue after applied discounts.",
            },
            {
                "name": "Total COGS",
                "table_name": "fact_orders",
                "user_formula": "SUM(fact_orders[cogs_usd])",
                "input_type": "dax",
                "description": "Total cost of goods sold.",
            },
            {
                "name": "Total Return Refunds",
                "table_name": "fact_returns",
                "user_formula": "SUM(fact_returns[refund_amount_usd])",
                "input_type": "dax",
                "description": "Total refund amounts processed for returned items.",
            },
        ]
        
        created_measures = []
        for m in measures_to_create:
            t0 = time.time()
            res = self.client.post("/api/measures/create", json=m)
            lat = round((time.time() - t0) * 1000, 2)
            logger.info(f"POST /api/measures/create ['{m['name']}'] -> HTTP {res.status_code} ({lat} ms)")
            if res.status_code != 200:
                self._record_error(phase, f"Failed to create measure '{m['name']}': {res.text}")
                return False
                
            data = res.json()
            meas = data.get("latest_measure") or data.get("measure", {})
            created_measures.append({
                "name": meas.get("name"),
                "dax": meas.get("dax_expression"),
                "sql": meas.get("sql_expression"),
                "verified": meas.get("verified"),
                "sample_value": meas.get("sample_value"),
            })
            logger.info(f"  Transpiled SQL: {meas.get('sql_expression')}")
            logger.info(f"  Dry-Run Verified: {meas.get('verified')} (Sample Value: {meas.get('sample_value')})")
            assert meas.get("verified"), f"Dry run failed for measure {m['name']}"
            
        self._dump_response("06_measures", created_measures)
        self.state_data["phase_metrics"][phase] = {
            "status": "PASS",
            "created_measures": created_measures,
        }
        return True

    # -------------------------------------------------------------------------
    # PHASE 7: Conversational BI & Analytical Queries ("Talk to Data")
    # -------------------------------------------------------------------------
    def run_phase7_conversational_queries(self) -> bool:
        phase = "phase7_conversational_queries"
        logger.info(">>> STARTING PHASE 7: Conversational BI ('Talk to Data') Analytical Execution")
        
        test_questions = [
            {
                "label": "Aggregation & Dimension Join (Region)",
                "question": "What is the total gross revenue by region name?",
            },
            {
                "label": "Top-N Ranking & Sorting (Sales Reps)",
                "question": "Who are the top 5 sales reps by total net revenue?",
            },
            {
                "label": "Category Breakdown (Products)",
                "question": "What is the total net revenue and COGS for each product category?",
            },
            {
                "label": "Multi-Fact Fan-Trap Query (Orders + Returns)",
                "question": "Compare total gross revenue and return refund amount by customer tier.",
            },
            {
                "label": "Filtered Aggregation (Completed Orders)",
                "question": "What is the total net revenue for completed orders by industry?",
            },
        ]
        
        query_results = []
        for item in test_questions:
            q = item["question"]
            label = item["label"]
            logger.info(f"Querying [{label}]: '{q}'")
            t0 = time.time()
            res = self.client.post("/api/chat/query", json={"question": q})
            lat = round((time.time() - t0) * 1000, 2)
            logger.info(f"POST /api/chat/query -> HTTP {res.status_code} ({lat} ms)")
            
            if res.status_code != 200:
                self._record_error(phase, f"Query failed for '{q}': HTTP {res.status_code}", res.text)
                return False
                
            data = res.json()
            entry = data.get("entry", {})
            resp = entry.get("response", {})
            rows = entry.get("rows", [])
            
            success = resp.get("success", False)
            sql = resp.get("executed_query", "")
            attempts = resp.get("attempts_used", 1)
            rec_chart = entry.get("chart_type", "")
            fan_trap = resp.get("fan_trap_mitigated", False)
            
            logger.info(f"  Success: {success} | Rows returned: {len(rows)} | Attempts used: {attempts}")
            logger.info(f"  Recommended Chart: {rec_chart}")
            logger.info(f"  Fan Trap Mitigated: {fan_trap}")
            logger.info(f"  Executed SQL:\n    {sql.replace(chr(10), ' ')}")
            
            if not success or len(rows) == 0:
                self._record_error(phase, f"Query returned unsuccessful or 0 rows: '{q}'", resp)
                return False
                
            query_results.append({
                "label": label,
                "question": q,
                "latency_ms": lat,
                "success": success,
                "row_count": len(rows),
                "attempts_used": attempts,
                "recommended_chart": rec_chart,
                "fan_trap_mitigated": fan_trap,
                "sample_rows": rows[:3],
                "executed_query": sql,
            })
            
        self._dump_response("07_chat_queries", query_results)
        self.state_data["phase_metrics"][phase] = {
            "status": "PASS",
            "queries_executed": len(query_results),
            "results": query_results,
        }
        return True

    # -------------------------------------------------------------------------
    # PHASE 8: Interactive Visualization & Chart Generation
    # -------------------------------------------------------------------------
    def run_phase8_chart_generation(self) -> bool:
        phase = "phase8_chart_generation"
        logger.info(">>> STARTING PHASE 8: Interactive Chart Generation & Plotly Verification")
        
        # Sample dataset from previous query
        sample_rows = [
            {"region_name": "North America", "total_gross_revenue": 1450200.0},
            {"region_name": "EMEA", "total_gross_revenue": 980400.0},
            {"region_name": "Asia Pacific", "total_gross_revenue": 620150.0},
            {"region_name": "Latin America", "total_gross_revenue": 210800.0},
        ]
        
        chart_types = ["bar", "pie", "horizontal_bar"]
        chart_results = []
        for ct in chart_types:
            req = {
                "rows": sample_rows,
                "chart_type": ct,
                "title": f"Regional Gross Revenue Breakdown ({ct.title()})",
            }
            t0 = time.time()
            res = self.client.post("/api/chat/chart", json=req)
            lat = round((time.time() - t0) * 1000, 2)
            logger.info(f"POST /api/chat/chart ['{ct}'] -> HTTP {res.status_code} ({lat} ms)")
            
            if res.status_code != 200:
                self._record_error(phase, f"Chart generation failed for '{ct}': {res.text}")
                return False
                
            data = res.json()
            raw_spec = data.get("plotly_spec", {})
            spec = json.loads(raw_spec) if isinstance(raw_spec, str) else raw_spec
            
            assert "data" in spec, f"Plotly spec missing 'data' key for {ct}"
            assert "layout" in spec, f"Plotly spec missing 'layout' key for {ct}"
            
            logger.info(f"  Chart '{ct}' rendered with {len(spec['data'])} trace(s) and title '{spec['layout'].get('title', {}).get('text', '')}'")
            chart_results.append({
                "chart_type": ct,
                "traces_count": len(spec["data"]),
                "latency_ms": lat,
            })
            
        self._dump_response("08_charts", chart_results)
        self.state_data["phase_metrics"][phase] = {
            "status": "PASS",
            "charts_tested": chart_results,
        }
        return True

    # -------------------------------------------------------------------------
    # PHASE 9: What-If Scenario Simulation & Root Cause
    # -------------------------------------------------------------------------
    def run_phase9_what_if_simulation(self) -> bool:
        phase = "phase9_what_if_simulation"
        logger.info(">>> STARTING PHASE 9: What-If Scenario Simulator & Dynamic Recalculation")
        
        sim_req = {
            "revenue_shock_pct": 20.0,  # +20% revenue expansion
            "cost_shock_pct": -10.0,   # -10% cost reduction
        }
        t0 = time.time()
        res = self.client.post("/api/dashboard/simulate", json=sim_req)
        lat = round((time.time() - t0) * 1000, 2)
        logger.info(f"POST /api/dashboard/simulate -> HTTP {res.status_code} ({lat} ms)")
        
        if res.status_code != 200:
            self._record_error(phase, f"Simulation returned HTTP {res.status_code}", res.text)
            return False
            
        sim_data = res.json()
        self._dump_response("09_what_if_simulation", sim_data)
        
        kpis = sim_data.get("kpis", {})
        baseline_rev = kpis.get("base_revenue", 0.0)
        simulated_rev = kpis.get("sim_revenue", 0.0)
        baseline_prof = kpis.get("base_profit", 0.0)
        simulated_prof = kpis.get("sim_profit", 0.0)
        
        logger.info(f"Baseline Revenue: ${baseline_rev:,.2f} -> Simulated: ${simulated_rev:,.2f}")
        logger.info(f"Baseline Profit: ${baseline_prof:,.2f} -> Simulated: ${simulated_prof:,.2f}")
        
        assert simulated_rev > baseline_rev, f"Simulated revenue should be higher with +20% shock (base: {baseline_rev}, sim: {simulated_rev})"
        assert simulated_prof > baseline_prof, f"Simulated profit should be higher with +20% rev and -10% cost (base: {baseline_prof}, sim: {simulated_prof})"
        
        self.state_data["phase_metrics"][phase] = {
            "status": "PASS",
            "latency_ms": lat,
            "baseline_revenue": baseline_rev,
            "simulated_revenue": simulated_rev,
            "baseline_profit": baseline_prof,
            "simulated_profit": simulated_prof,
        }
        return True

    # -------------------------------------------------------------------------
    # PHASE 10: Executive Pinboard & Report Generation
    # -------------------------------------------------------------------------
    def run_phase10_pinboard_and_export(self) -> bool:
        phase = "phase10_pinboard_and_export"
        logger.info(">>> STARTING PHASE 10: Executive Pinboard & Boardroom Report Export")
        
        # 1. Pin a widget to dashboard
        pin_req = {
            "title": "Regional Revenue Breakdown",
            "user_question": "What is the total gross revenue by region name?",
            "executed_query": "SELECT dim_regions.region_name, SUM(fact_orders.gross_revenue_usd) AS total_gross_revenue FROM fact_orders LEFT JOIN dim_customers ON fact_orders.customer_id = dim_customers.customer_id LEFT JOIN dim_regions ON dim_customers.region_id = dim_regions.region_id GROUP BY dim_regions.region_name",
            "query_dialect": "duckdb",
            "chart_type": "bar",
        }
        res_pin = self.client.post("/api/dashboard/pin", json=pin_req)
        logger.info(f"POST /api/dashboard/pin -> HTTP {res_pin.status_code}")
        if res_pin.status_code != 200:
            self._record_error(phase, f"Pinning widget failed: {res_pin.text}")
            return False
            
        # 2. Export HTML executive report
        t0 = time.time()
        res_report = self.client.get("/api/export/report")
        lat_rep = round((time.time() - t0) * 1000, 2)
        logger.info(f"GET /api/export/report -> HTTP {res_report.status_code} ({lat_rep} ms)")
        if res_report.status_code != 200:
            self._record_error(phase, f"Report export failed: {res_report.text}")
            return False
        assert "<!DOCTYPE html>" in res_report.text or "<html" in res_report.text.lower(), "Report HTML content invalid"
        assert len(res_report.text) > 500, "Report HTML content is too short"
        
        report_file = LOG_DIR / "executive_report.html"
        report_file.write_text(res_report.text, encoding="utf-8")
        logger.info(f"Executive Report saved to {report_file} ({len(res_report.text)} bytes)")
        
        # 3. Export OKF YAML
        res_okf = self.client.get("/api/export/okf")
        logger.info(f"GET /api/export/okf -> HTTP {res_okf.status_code}")
        assert res_okf.status_code == 200
        
        # 4. Export DAX script
        res_dax = self.client.get("/api/export/dax")
        logger.info(f"GET /api/export/dax -> HTTP {res_dax.status_code}")
        assert res_dax.status_code == 200
        
        self.state_data["phase_metrics"][phase] = {
            "status": "PASS",
            "report_bytes": len(res_report.text),
            "okf_bytes": len(res_okf.text),
            "dax_bytes": len(res_dax.text),
        }
        return True

    # -------------------------------------------------------------------------
    # PHASE 11: Security Sandbox Probing
    # -------------------------------------------------------------------------
    def run_phase11_security_firewall(self) -> bool:
        phase = "phase11_security_firewall"
        logger.info(">>> STARTING PHASE 11: Security Firewall & Read-Only AST Defense Verification")
        
        forbidden_payloads = [
            "DROP TABLE fact_orders",
            "DELETE FROM dim_customers WHERE customer_id > 1000",
            "UPDATE fact_orders SET gross_revenue_usd = 0",
            "SELECT * FROM read_csv('/etc/passwd')",
            "COPY fact_orders TO 'exfil.csv'",
        ]
        
        blocked_count = 0
        for payload in forbidden_payloads:
            t0 = time.time()
            res = self.client.post("/api/sql/execute", json={"query": payload})
            lat = round((time.time() - t0) * 1000, 2)
            logger.info(f"POST /api/sql/execute ['{payload[:30]}...'] -> HTTP {res.status_code} ({lat} ms)")
            if res.status_code == 400:
                blocked_count += 1
                logger.info(f"  BLOCKED as expected: {res.json().get('detail', '')}")
            else:
                self._record_error(phase, f"Security Breach: payload '{payload}' was NOT blocked (got {res.status_code})")
                return False
                
        # Legitimate analytical read-only query
        legit_query = "SELECT category, count(*) as n FROM dim_products GROUP BY category"
        res_legit = self.client.post("/api/sql/execute", json={"query": legit_query})
        logger.info(f"POST /api/sql/execute [Legitimate Query] -> HTTP {res_legit.status_code}")
        assert res_legit.status_code == 200, f"Legitimate query blocked: {res_legit.text}"
        
        self.state_data["phase_metrics"][phase] = {
            "status": "PASS",
            "blocked_payloads_count": blocked_count,
            "legitimate_query_passed": True,
        }
        return True

    # -------------------------------------------------------------------------
    # Main Orchestrator Loop (Interval Execution & Checkpoint Management)
    # -------------------------------------------------------------------------
    def execute_all_phases(self, resume: bool = True) -> bool:
        logger.info("=================================================================")
        logger.info("  DATA_TALK BLACK-BOX TESTING SUITE — INTERVAL EXECUTION ENGINE  ")
        logger.info("=================================================================")
        self.init_client()
        
        for phase_name in self.PHASES:
            if resume and phase_name in self.state_data.get("completed_phases", []):
                logger.info(f"[SKIPPING] Phase '{phase_name}' already completed in previous checkpoint.")
                continue
                
            self.state_data["current_phase"] = phase_name
            self._save_checkpoint()
            
            method_name = f"run_{phase_name}"
            method = getattr(self, method_name, None)
            if not method:
                logger.error(f"No runner found for phase {phase_name}")
                return False
                
            success = method()
            if not success:
                logger.error(f"Execution halted at phase '{phase_name}'. Checkpoint saved. Run with --resume to continue.")
                return False
                
            if phase_name not in self.state_data["completed_phases"]:
                self.state_data["completed_phases"].append(phase_name)
            self._save_checkpoint()
            logger.info(f"[CHECKPOINT] Phase '{phase_name}' successfully completed and checkpointed.\n")
            
        logger.info("=================================================================")
        logger.info("  ALL BLACK-BOX TESTING PHASES COMPLETED WITH 100% SUCCESS RATE  ")
        logger.info("=================================================================")
        return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data_Talk Black-Box Test Harness")
    parser.add_argument("--no-resume", action="store_true", help="Start from beginning without resuming")
    parser.add_argument("--phase", type=str, default="", help="Run a specific phase only")
    args = parser.parse_args()
    
    runner = BlackBoxRunner()
    if args.no_resume:
        runner.state_data = {
            "completed_phases": [],
            "current_phase": "phase1_health",
            "phase_metrics": {},
            "errors": [],
            "last_updated": datetime.now().isoformat(),
        }
        runner._save_checkpoint()
        
    if args.phase:
        runner.init_client()
        method = getattr(runner, f"run_{args.phase}", None)
        if method:
            ok = method()
            sys.exit(0 if ok else 1)
        else:
            print(f"Unknown phase: {args.phase}")
            sys.exit(1)
            
    success = runner.execute_all_phases(resume=not args.no_resume)
    sys.exit(0 if success else 1)
