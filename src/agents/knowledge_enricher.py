"""Autonomous Business Knowledge & Semantic Domain Enricher for Data_Talk.

Dramatically increases model awareness, domain semantics, and analytical depth by:
1. Identifying the business domain (E-Commerce/Retail, University/HigherEd, SaaS/Cloud,
   Healthcare, Financial Services, Supply Chain, or General Enterprise).
2. Generating rich business names, operational descriptions, and domain tags for tables and columns.
3. Automatically deciphering cryptic status codes (e.g., 'C' -> Completed, 'X' -> Cancelled, 'P' -> Pending).
4. Generating standard domain KPI measures (SQL + DAX) and business governance rules.
5. Indexing the enriched semantic domain knowledge into the Hybrid RAG store.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any

from src.knowledge.okf_models import (
    BusinessRule,
    ColumnProfile,
    OKFWorkspace,
    SemanticMeasure,
    TableRole,
    TableSchema,
)

if TYPE_CHECKING:
    from src.agents.llm_router import LLMRouter
    from src.knowledge.rag_store import RAGStore

_LOGGER = logging.getLogger(__name__)


DOMAIN_CATALOG = {
    "ecommerce": {
        "title": "Retail & E-Commerce",
        "industry": "Commercial Sales & Retail Operations",
        "keywords": {
            "order", "orders", "product", "products", "customer", "customers", "client",
            "return", "returns", "refund", "sales", "revenue", "cogs", "sku", "cart",
            "checkout", "inventory", "disc_amt", "discount", "ord_stat", "sales_channel",
        },
        "table_hints": {
            "orders": {
                "business_name": "Customer Orders & Sales Transactions",
                "description": "Records individual customer purchases, line items, order status ('C'=Completed, 'X'=Cancelled), revenue, and direct COGS.",
            },
            "products": {
                "business_name": "Product Master Catalog",
                "description": "Stores master SKU metadata, product category classifications, list prices, and unit manufacturing costs.",
            },
            "customers": {
                "business_name": "Customer Accounts & Demographics",
                "description": "Contains customer master profiles, relationship tiers (Enterprise, Mid-Market, SMB), geographic region codes, and contact info.",
            },
            "returns": {
                "business_name": "Merchandise Returns & Refunds",
                "description": "Logs customer returned merchandise, reason codes (defective, wrong item, dissatisfied), and refunded amounts.",
            },
        },
        "column_hints": {
            "order_id": ("Order ID", "Unique primary transaction identifier for each customer order"),
            "order_date": ("Order Placement Date", "Timestamp when the order was placed by the customer"),
            "customer_id": ("Customer Identifier", "Foreign key linking the order to customer master profiles"),
            "product_id": ("Product SKU ID", "Foreign key linking the transaction to product master records"),
            "rev_gross_lcy": ("Gross Revenue (LCY)", "Pre-discount gross transaction revenue in local currency"),
            "rev_net_usd": ("Net Recognized Revenue (USD)", "Net revenue recognized in USD after discounts and promotional adjustments"),
            "disc_amt": ("Discount Amount", "Total promotional or discretionary discount amount applied to the order"),
            "cogs_usd": ("Cost of Goods Sold (USD)", "Direct production and procurement cost associated with the order"),
            "ord_stat": ("Order Status Code", "Lifecycle state: 'C'=Completed, 'P'=Pending, 'X'=Cancelled, 'H'=On Hold"),
            "sales_channel": ("Sales Channel", "Go-to-market channel (Direct Web, Partner/Distributor, Retail Store, Mobile App)"),
            "cust_tier": ("Customer Tier", "Commercial customer account classification: Enterprise, Mid-Market, or SMB"),
            "region_code": ("Geographic Region", "Sales territory or operational region code (e.g. NA, EMEA, APAC, LATAM)"),
            "ret_reason_cd": ("Return Reason Code", "Root cause classification for customer returns (DEF=Defective, WNG=Wrong Item, UNS=Dissatisfied)"),
            "refund_amt_usd": ("Refund Amount (USD)", "Total dollar amount refunded to customer upon merchandise return"),
        },
        "code_meanings": {
            "ord_stat": {"C": "Completed", "P": "Pending", "X": "Cancelled", "H": "On Hold"},
            "cust_tier": {"ENT": "Enterprise", "MID": "Mid-Market", "SMB": "Small Business"},
            "ret_reason_cd": {
                "DEF": "Defective Merchandise",
                "WNG": "Wrong Item Shipped",
                "UNS": "Customer Dissatisfied",
                "LATE": "Late Delivery",
            },
        },
        "standard_measures": [
            {
                "name": "Net Revenue",
                "table_name": "Metrics",
                "description": "Total recognized revenue excluding cancelled orders",
                "sql": "SUM(orders.rev_net_usd)",
                "dax": "CALCULATE(SUM(orders[rev_net_usd]), orders[ord_stat] <> \"X\")",
                "involved": ["orders"],
            },
            {
                "name": "Order Count",
                "table_name": "Metrics",
                "description": "Total volume of completed customer orders",
                "sql": "COUNT(orders.order_id)",
                "dax": "CALCULATE(DISTINCTCOUNT(orders[order_id]), orders[ord_stat] <> \"X\")",
                "involved": ["orders"],
            },
            {
                "name": "Average Order Value (AOV)",
                "table_name": "Metrics",
                "description": "Average net revenue generated per completed order transaction",
                "sql": "SUM(orders.rev_net_usd) / NULLIF(COUNT(orders.order_id), 0)",
                "dax": "DIVIDE([Net Revenue], [Order Count])",
                "involved": ["orders"],
            },
            {
                "name": "Total Profit",
                "table_name": "Metrics",
                "description": "Gross margin contribution (Net Revenue minus COGS)",
                "sql": "SUM(orders.rev_net_usd) - SUM(orders.cogs_usd)",
                "dax": "CALCULATE(SUM(orders[rev_net_usd]) - SUM(orders[cogs_usd]), orders[ord_stat] <> \"X\")",
                "involved": ["orders"],
            },
        ],
        "business_rules": [
            {
                "term": "Exclude Cancelled Orders",
                "definition": "Orders with status code 'X' represent cancelled or voided transactions and must be excluded from all recognized revenue, sales, and order counts.",
                "sql_filter": "orders.ord_stat <> 'X'",
                "dax_filter": "orders[ord_stat] <> \"X\"",
            },
            {
                "term": "Net vs Gross Revenue",
                "definition": "Standard executive 'Revenue' or 'Sales' queries refer to Net Recognized Revenue in USD (`rev_net_usd`). Gross revenue should only be reported when explicitly requested.",
                "sql_filter": "",
                "dax_filter": "",
            },
        ],
    },
    "education": {
        "title": "Higher Education & Academics",
        "industry": "University & Academic Administration",
        "keywords": {
            "student", "students", "college", "colleges", "department", "departments",
            "marks", "grade", "grades", "enroll", "enrollment", "enrollments",
            "course", "courses", "faculty", "professor", "tuition", "gpa", "gender",
            "semester", "academic", "degree", "campus",
        },
        "table_hints": {
            "students": {
                "business_name": "Student Demographic Master",
                "description": "Master registry of admitted students, including demographic profiles, gender, admitted college, and degree level.",
            },
            "colleges": {
                "business_name": "Colleges & Academic Schools",
                "description": "Constituent schools, colleges, and academic divisions within the university.",
            },
            "departments": {
                "business_name": "Academic Departments",
                "description": "Academic faculty departments (e.g. Computer Science, Mechanical Engineering, Economics).",
            },
            "enrollments": {
                "business_name": "Course Enrollments & Grades",
                "description": "Student semester registrations, course attendance, and examination marks/grades.",
            },
            "enrollments_marks": {
                "business_name": "Student Exam Performance & Marks",
                "description": "Detailed academic evaluation records tracking total examination marks and grading benchmarks.",
            },
        },
        "column_hints": {
            "student_id": ("Student ID", "Unique identification number assigned to each enrolled student"),
            "college_id": ("College Code", "Unique identifier for the affiliated university college"),
            "college_name": ("College Name", "Full official name of the academic college or school"),
            "department_id": ("Department ID", "Identifier of the student's primary academic department"),
            "department_name": ("Department Name", "Name of the academic department (e.g. Electrical Engineering)"),
            "gender": ("Gender", "Student gender identity (M=Male, F=Female, O=Other)"),
            "total_marks": ("Total Examination Marks", "Cumulative examination score obtained across evaluated courses"),
            "marks_obtained": ("Marks Obtained", "Raw examination score achieved by the student"),
        },
        "code_meanings": {
            "gender": {"M": "Male", "F": "Female", "O": "Other", "m": "Male", "f": "Female"},
        },
        "standard_measures": [
            {
                "name": "Total Students",
                "table_name": "Metrics",
                "description": "Total count of active enrolled students",
                "sql": "COUNT(DISTINCT students.student_id)",
                "dax": "DISTINCTCOUNT(students[student_id])",
                "involved": ["students"],
            },
            {
                "name": "Average Examination Marks",
                "table_name": "Metrics",
                "description": "Average total examination marks attained across students",
                "sql": "ROUND(AVG(enrollments_marks.total_marks), 2)",
                "dax": "AVERAGE(enrollments_marks[total_marks])",
                "involved": ["enrollments_marks"],
            },
        ],
        "business_rules": [
            {
                "term": "Gender Diversity Ratio",
                "definition": "Gender ratio calculations should compare male, female, and other student counts as a percentage of total student enrollment within each college or department.",
                "sql_filter": "",
                "dax_filter": "",
            },
        ],
    },
    "saas": {
        "title": "SaaS & Subscription Cloud",
        "industry": "Software-as-a-Service & Recurring Revenue",
        "keywords": {
            "subscription", "subscriptions", "plan", "mrr", "arr", "churn", "seats",
            "user", "users", "account", "accounts", "trial", "license", "renewal", "usage",
        },
        "table_hints": {
            "subscriptions": {
                "business_name": "Customer Subscriptions & Contracts",
                "description": "Subscription contracts, pricing plan tiers, billing cycles, Monthly Recurring Revenue (MRR), and renewal dates.",
            },
            "accounts": {
                "business_name": "Organization Accounts",
                "description": "Master account profiles, billing entities, industry segments, and account health scores.",
            },
            "users": {
                "business_name": "Licensed Seat Users",
                "description": "Active licensed users, last login timestamps, and platform engagement telemetry.",
            },
        },
        "column_hints": {
            "mrr": ("Monthly Recurring Revenue ($)", "Contracted normalized monthly recurring revenue"),
            "arr": ("Annual Recurring Revenue ($)", "Annualized contract value of the active subscription"),
            "status": ("Subscription Status", "Contract state ('active', 'churned', 'past_due', 'trial')"),
            "seats": ("Licensed Seats", "Number of provisioned user seats allocated to the account"),
        },
        "code_meanings": {
            "status": {"A": "Active", "C": "Churned", "T": "Trial", "P": "Past Due"},
        },
        "standard_measures": [
            {
                "name": "Total ARR",
                "table_name": "Metrics",
                "description": "Total Annual Recurring Revenue across all active subscriptions",
                "sql": "SUM(subscriptions.arr)",
                "dax": "CALCULATE(SUM(subscriptions[arr]), subscriptions[status] = \"active\")",
                "involved": ["subscriptions"],
            },
        ],
        "business_rules": [],
    },
}


class BusinessKnowledgeEnricher:
    """Intelligent semantic knowledge enricher for OKF schemas and RAG retrieval."""

    @classmethod
    def detect_domain(cls, workspace: OKFWorkspace) -> dict[str, Any]:
        """Detect the primary business domain based on table names, column names, and sample values."""
        if workspace.domain_title:
            return {
                "domain_key": workspace.domain_tag or "custom",
                "domain_title": workspace.domain_title,
                "industry": workspace.domain_title,
                "confidence": 1.0,
                "matched_terms": [],
                "summary": f"Configured business domain: {workspace.domain_title}.",
                "is_user_override": True,
            }

        active_tables = workspace.active_tables()
        if not active_tables:
            return {
                "domain_key": "general",
                "domain_title": "Enterprise Operations & Analytics",
                "industry": "General Business",
                "confidence": 0.5,
                "summary": "General organizational dataset.",
            }

        all_text = []
        for t in active_tables:
            all_text.append(t.name.lower())
            for c in t.columns:
                all_text.append(c.name.lower())
                for s in c.sample_values[:3]:
                    all_text.append(str(s).lower())

        text_blob = " ".join(all_text)
        tokens = set(re.findall(r"[a-z0-9_]+", text_blob))

        best_domain = "general"
        best_score = 0
        best_details = None

        for d_key, d_info in DOMAIN_CATALOG.items():
            matches = tokens.intersection(d_info["keywords"])
            score = len(matches)
            if score > best_score:
                best_score = score
                best_domain = d_key
                best_details = d_info

        if best_score >= 2 and best_details:
            confidence = min(0.98, 0.45 + (best_score * 0.08))
            return {
                "domain_key": best_domain,
                "domain_title": best_details["title"],
                "industry": best_details["industry"],
                "confidence": round(confidence, 2),
                "matched_terms": sorted(tokens.intersection(best_details["keywords"])),
                "summary": f"Identified as {best_details['title']} ({best_details['industry']}) based on detected entities.",
            }

        return {
            "domain_key": "general",
            "domain_title": "Enterprise Operations & Analytics",
            "industry": "General Business",
            "confidence": 0.5,
            "matched_terms": [],
            "summary": "Multi-entity business dataset with structured tables and attributes.",
        }

    @classmethod
    def auto_enrich_workspace(
        cls,
        workspace: OKFWorkspace,
        llm_router: LLMRouter | None = None,
        rag_store: RAGStore | None = None,
    ) -> dict[str, Any]:
        """Enrich table descriptions, column aliases, value meanings, measures, and business rules."""
        domain_info = cls.detect_domain(workspace)
        domain_key = domain_info["domain_key"]
        domain_preset = DOMAIN_CATALOG.get(domain_key, {})

        workspace.domain_tag = domain_key
        workspace.domain_title = domain_info.get("domain_title", "Enterprise Operations")

        tables_enriched = 0
        columns_enriched = 0
        rules_added = 0
        measures_added = 0

        table_hints = domain_preset.get("table_hints", {})
        column_hints = domain_preset.get("column_hints", {})
        code_meanings = domain_preset.get("code_meanings", {})

        # 1. Enrich Table Descriptions and Business Names
        for tbl in workspace.active_tables():
            t_low = tbl.name.lower()
            matched_hint = None
            for hint_k, hint_v in table_hints.items():
                if hint_k == t_low or hint_k in t_low:
                    matched_hint = hint_v
                    break

            if matched_hint:
                if not tbl.business_name or tbl.business_name == tbl.name:
                    tbl.business_name = matched_hint["business_name"]
                    tables_enriched += 1
                if not tbl.description or tbl.description in ("Core data table", ""):
                    tbl.description = matched_hint["description"]
            else:
                if not tbl.business_name:
                    tbl.business_name = tbl.name.replace("_", " ").title()
                if not tbl.description:
                    role_desc = "Transactional event ledger" if tbl.table_role == TableRole.FACT else "Master entity dimension"
                    tbl.description = f"{role_desc} containing {len(tbl.columns)} attributes and {tbl.row_count:,} records."

            # 2. Enrich Columns: Aliases, Descriptions & Code Meanings
            for col in tbl.columns:
                c_low = col.name.lower()
                c_hint = column_hints.get(c_low)
                if c_hint:
                    if not col.business_alias:
                        col.business_alias = c_hint[0]
                        columns_enriched += 1
                    if not col.description:
                        col.description = c_hint[1]
                else:
                    if not col.business_alias:
                        clean_alias = c_low.replace("_id", " ID").replace("_", " ").title()
                        col.business_alias = clean_alias

                # Check for low-cardinality status codes
                if c_low in code_meanings and not col.value_meanings:
                    col.value_meanings = dict(code_meanings[c_low])
                elif not col.value_meanings and 1 < len(col.sample_values) <= 6:
                    # Auto-infer common binary / status codes
                    val_set = {str(v).strip().upper() for v in col.sample_values}
                    if val_set.issubset({"C", "P", "X", "H"}):
                        col.value_meanings = {"C": "Completed", "P": "Pending", "X": "Cancelled", "H": "On Hold"}
                    elif val_set.issubset({"M", "F", "O"}):
                        col.value_meanings = {"M": "Male", "F": "Female", "O": "Other"}
                    elif val_set.issubset({"Y", "N"}):
                        col.value_meanings = {"Y": "Yes / True", "N": "No / False"}
                    elif val_set.issubset({"A", "I"}):
                        col.value_meanings = {"A": "Active", "I": "Inactive"}

        # 3. Add Domain Business Rules if not already present
        existing_rule_terms = {r.term.lower() for r in workspace.business_rules}
        for br in domain_preset.get("business_rules", []):
            if br["term"].lower() not in existing_rule_terms:
                rid = f"rule_{len(workspace.business_rules) + 1}"
                new_rule = BusinessRule(
                    id=rid,
                    category="domain_governance",
                    term=br["term"],
                    definition=br["definition"],
                    sql_filter_hint=br.get("sql_filter", ""),
                    dax_filter_hint=br.get("dax_filter", ""),
                    source="auto_enricher",
                )
                workspace.business_rules.append(new_rule)
                rules_added += 1

        # 4. Add Standard Domain Measures if applicable
        existing_measure_names = {m.name.lower() for m in workspace.measures}
        for sm in domain_preset.get("standard_measures", []):
            if sm["name"].lower() not in existing_measure_names:
                # Check if involved tables exist in workspace
                all_tables_exist = all(workspace.get_table(t) is not None for t in sm["involved"])
                if all_tables_exist:
                    mid = f"m_{sm['name'].lower().replace(' ', '_').replace('%', 'pct')}"
                    new_meas = SemanticMeasure(
                        id=mid,
                        name=sm["name"],
                        table_name=sm.get("table_name", "Metrics"),
                        description=sm["description"],
                        dax_expression=sm["dax"],
                        sql_expression=sm["sql"],
                        involved_tables=sm["involved"],
                        verified=True,
                    )
                    workspace.measures.append(new_meas)
                    measures_added += 1

        # 5. Sync Enriched Semantic Catalog to RAG
        if rag_store is not None:
            cls.sync_enriched_knowledge_to_rag(workspace, rag_store, domain_info)

        return {
            "domain": domain_info,
            "tables_enriched": tables_enriched,
            "columns_enriched": columns_enriched,
            "rules_added": rules_added,
            "measures_added": measures_added,
        }

    @classmethod
    def sync_enriched_knowledge_to_rag(
        cls,
        workspace: OKFWorkspace,
        rag_store: RAGStore,
        domain_info: dict[str, Any],
    ) -> None:
        """Deeply indexes the enriched domain context into the RAG store."""
        # 1. Index domain overview
        rag_store.upsert(
            collection="business_rules",
            doc_id="domain_context_overview",
            content=(
                f"Domain: {domain_info.get('domain_title', 'Enterprise')}\n"
                f"Industry: {domain_info.get('industry', 'Business')}\n"
                f"Overview: {domain_info.get('summary', '')}"
            ),
            metadata={"type": "domain_overview", "domain": domain_info.get("domain_key", "general")},
        )

        # 2. Index each business rule
        for rule in workspace.business_rules:
            rag_store.upsert(
                collection="business_rules",
                doc_id=rule.id,
                content=f"Term: {rule.term}\nDefinition: {rule.definition}\nSQL Filter: {rule.sql_filter_hint}",
                metadata={"term": rule.term, "sql_filter_hint": rule.sql_filter_hint},
            )

        # 3. Index each table with full semantic aliases and value meanings
        for tbl in workspace.active_tables():
            col_lines = []
            for c in tbl.columns:
                if c.is_blocked:
                    continue
                alias_str = f" [Alias: {c.business_alias}]" if c.business_alias else ""
                desc_str = f" - {c.description}" if c.description else ""
                codes_str = f" (Meanings: {c.value_meanings})" if c.value_meanings else ""
                col_lines.append(f"  * `{c.name}` ({c.data_type}){alias_str}{desc_str}{codes_str}")

            content = (
                f"Table `{tbl.name}`: {tbl.business_name or tbl.name}\n"
                f"Role: {tbl.table_role.value} | Rows: {tbl.row_count:,}\n"
                f"Description: {tbl.description}\n"
                f"Columns & Business Semantics:\n" + "\n".join(col_lines)
            )
            rag_store.upsert(
                collection="schema_glossary",
                doc_id=f"enriched_table_{tbl.name}",
                content=content,
                metadata={"table_name": tbl.name, "business_name": tbl.business_name},
            )
