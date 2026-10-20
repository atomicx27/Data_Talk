"""Unified Semantic Measure & Calculated Column Studio Compiler (DAX <-> SQL).

Enables Power BI-grade Measures & Calculated Columns across ALL data sources:
- Users can input **DAX** (e.g., `DIVIDE(SUM(orders[profit_usd]), SUM(orders[rev_net_usd]), 0)`),
  **SQL**, or **Natural Language** (e.g., `"Net Revenue after subtracting refunds"`).
- Deterministically compiles common DAX idioms (`SUM`, `AVERAGE`, `MIN`, `MAX`, `COUNT`, `DISTINCTCOUNT`,
  `DIVIDE`, `CALCULATE(metric, filter)`, `IF`) into SQL expressions—and falls back to `LiteLLM` for complex time-intelligence.
- Runs a **Live Dry-Run Validation Query** against the connected data source so every saved measure is verified.
- Indexes every user-provided DAX measure into the `RAGStore` (`few_shot_queries`) so the AI learns the user's DAX patterns.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from src.agents.llm_router import LLMRouter
from src.knowledge.okf_models import (
    CalculatedColumn,
    OKFWorkspace,
    SemanticMeasure,
)
from src.knowledge.rag_store import RAGStore

if TYPE_CHECKING:
    from src.connectors.base import BaseConnector


class MeasureCompiler:
    """Compiles, cross-translates, and dry-run validates DAX & SQL Measures and Calculated Columns."""

    STARTER_MEASURE_TEMPLATES: list[dict[str, str]] = [
        {
            "name": "Recognized Net Revenue",
            "table_name": "orders",
            "description": "Total Net Revenue in USD excluding Cancelled ('X') orders.",
            "dax_expression": "CALCULATE(SUM(orders[rev_net_usd]), orders[ord_stat] <> \"X\")",
            "sql_expression": "SUM(CASE WHEN orders.ord_stat <> 'X' THEN orders.rev_net_usd ELSE 0 END)",
        },
        {
            "name": "Gross Margin %",
            "table_name": "orders",
            "description": "Profit divided by Net Revenue across active orders.",
            "dax_expression": "DIVIDE(SUM(orders[profit_usd]), SUM(orders[rev_net_usd]), 0)",
            "sql_expression": "COALESCE(SUM(orders.profit_usd) / NULLIF(SUM(orders.rev_net_usd), 0), 0)",
        },
        {
            "name": "Average Order Value (AOV)",
            "table_name": "orders",
            "description": "Recognized Net Revenue divided by distinct completed/pending orders.",
            "dax_expression": "DIVIDE(CALCULATE(SUM(orders[rev_net_usd]), orders[ord_stat] <> \"X\"), DISTINCTCOUNT(orders[order_id]), 0)",
            "sql_expression": "COALESCE(SUM(CASE WHEN orders.ord_stat <> 'X' THEN orders.rev_net_usd ELSE 0 END) / NULLIF(COUNT(DISTINCT orders.order_id), 0), 0)",
        },
        {
            "name": "Return Rate % (Revenue)",
            "table_name": "returns",
            "description": "Total Refunded USD divided by Total Net Order Revenue.",
            "dax_expression": "DIVIDE(SUM(returns[refund_amt_usd]), SUM(orders[rev_net_usd]), 0)",
            "sql_expression": "COALESCE(SUM(returns.refund_amt_usd) / NULLIF(SUM(orders.rev_net_usd), 0), 0)",
        },
    ]

    # ------------------------------------------------------------------
    # Deterministic DAX <-> SQL Transpiler for Standard BI Idioms
    # ------------------------------------------------------------------
    @classmethod
    def transpile_dax_to_sql(
        cls,
        dax_expr: str,
        default_table: str = "orders",
        workspace: OKFWorkspace | None = None,
        llm_router: LLMRouter | None = None,
    ) -> str:
        """Compile a DAX measure or column formula into an equivalent SQL aggregation expression."""
        cleaned = dax_expr.strip()
        # Strip leading `MeasureName = ` if the user pasted a full DAX assignment
        if "=" in cleaned and not cleaned.upper().startswith(("IF(", "CALCULATE(", "DIVIDE(", "SUM(", "AVERAGE(", "COUNT")):
            lhs, rhs = cleaned.split("=", 1)
            if len(lhs.strip()) < 50 and "(" not in lhs:
                cleaned = rhs.strip()

        # Pattern 1: CALCULATE(SUM(table[col]), table[filter_col] op "val")
        calc_match = re.match(
            r"^CALCULATE\(\s*(SUM|AVERAGE|AVG|COUNT|MIN|MAX)\(\s*'?([A-Za-z0-9_]+)'?\[([A-Za-z0-9_]+)\]\s*\)\s*,\s*'?([A-Za-z0-9_]+)'?\[([A-Za-z0-9_]+)\]\s*(=|<>|!=|>|<|>=|<=)\s*[\"']?([^\"')]+)[\"']?\s*\)$",
            cleaned,
            re.IGNORECASE,
        )
        if calc_match:
            agg, tbl, col, f_tbl, f_col, op, val = calc_match.groups()
            sql_agg = "AVG" if agg.upper() == "AVERAGE" else agg.upper()
            sql_op = "<>" if op == "!=" else op
            val_fmt = val if val.replace(".", "", 1).isdigit() else f"'{val}'"
            else_branch = "0" if sql_agg == "SUM" else "NULL"
            return f"{sql_agg}(CASE WHEN {f_tbl}.{f_col} {sql_op} {val_fmt} THEN {tbl}.{col} ELSE {else_branch} END)"

        # Pattern 2: DIVIDE(numerator, denominator, [alt])
        if cleaned.upper().startswith("DIVIDE(") and cleaned.endswith(")"):
            inner = cleaned[7:-1]
            parts = cls._split_top_level_args(inner)
            if len(parts) >= 2:
                num_sql = cls.transpile_dax_to_sql(parts[0], default_table, workspace, llm_router)
                den_sql = cls.transpile_dax_to_sql(parts[1], default_table, workspace, llm_router)
                alt_sql = parts[2].strip() if len(parts) >= 3 else "0"
                return f"COALESCE(({num_sql}) / NULLIF(({den_sql}), 0), {alt_sql})"

        # Pattern 3: DISTINCTCOUNT(table[col])
        dc_match = re.match(
            r"^DISTINCTCOUNT\(\s*'?([A-Za-z0-9_]+)'?\[([A-Za-z0-9_]+)\]\s*\)$",
            cleaned,
            re.IGNORECASE,
        )
        if dc_match:
            tbl, col = dc_match.groups()
            return f"COUNT(DISTINCT {tbl}.{col})"

        # Pattern 4: Standard Aggregations SUM/AVERAGE/MIN/MAX/COUNT(table[col])
        agg_match = re.match(
            r"^(SUM|AVERAGE|AVG|MIN|MAX|COUNT)\(\s*'?([A-Za-z0-9_]+)'?\[([A-Za-z0-9_]+)\]\s*\)$",
            cleaned,
            re.IGNORECASE,
        )
        if agg_match:
            agg, tbl, col = agg_match.groups()
            sql_agg = "AVG" if agg.upper() == "AVERAGE" else agg.upper()
            return f"{sql_agg}({tbl}.{col})"

        # Pattern 5: Row-level IF(cond, true_val, false_val) for Calculated Columns
        if cleaned.upper().startswith("IF(") and cleaned.endswith(")"):
            inner = cleaned[3:-1]
            parts = cls._split_top_level_args(inner)
            if len(parts) == 3:
                cond = cls._replace_dax_col_refs(parts[0], default_table).replace('"', "'")
                t_val = cls._replace_dax_col_refs(parts[1], default_table).replace('"', "'")
                f_val = cls._replace_dax_col_refs(parts[2], default_table).replace('"', "'")
                return f"CASE WHEN {cond} THEN {t_val} ELSE {f_val} END"

        # Pattern 6: General arithmetic with table[col] or [col] references
        replaced = cls._replace_dax_col_refs(cleaned, default_table)
        replaced = re.sub(r"\bAVERAGE\s*\(", "AVG(", replaced, flags=re.IGNORECASE)
        replaced = re.sub(r"\bDISTINCTCOUNT\s*\(([^)]+)\)", r"COUNT(DISTINCT \1)", replaced, flags=re.IGNORECASE)

        # If LLM router is configured and complex DAX keywords remain (like SAMEPERIODLASTYEAR / FILTER), ask LLM
        if (
            llm_router is not None
            and llm_router.is_configured()
            and any(k in replaced.upper() for k in ("CALCULATE", "FILTER", "ALL", "SAMEPERIODLASTYEAR", "DATEADD"))
        ):
            try:
                resp = llm_router.complete_json(
                    purpose="dax_to_sql_transpile",
                    system_prompt=(
                        "You are a senior BI compiler. Convert the given DAX expression into a single valid SQL "
                        "aggregation expression using table.column notation. Return JSON: {\"sql_expression\": \"...\"}"
                    ),
                    user_prompt=f"Default table: {default_table}\nDAX: {cleaned}",
                )
                if resp.get("sql_expression"):
                    return str(resp["sql_expression"])
            except Exception:
                pass

        return replaced

    @classmethod
    def transpile_sql_to_dax(
        cls,
        sql_expr: str,
        default_table: str = "orders",
    ) -> str:
        """Convert a SQL aggregation or column expression into canonical Power BI DAX."""
        cleaned = sql_expr.strip()
        # Convert COUNT(DISTINCT tbl.col) -> DISTINCTCOUNT(tbl[col])
        dax = re.sub(
            r"COUNT\s*\(\s*DISTINCT\s+([A-Za-z0-9_]+)\.([A-Za-z0-9_]+)\s*\)",
            r"DISTINCTCOUNT(\1[\2])",
            cleaned,
            flags=re.IGNORECASE,
        )
        # Convert AVG(tbl.col) -> AVERAGE(tbl[col])
        dax = re.sub(r"\bAVG\s*\(", "AVERAGE(", dax, flags=re.IGNORECASE)
        # Convert tbl.col -> tbl[col]
        dax = re.sub(r"\b([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)\b", r"\1[\2]", dax)
        return dax

    @staticmethod
    def _replace_dax_col_refs(expr: str, default_table: str) -> str:
        # Convert 'table'[col] or table[col] -> table.col
        step1 = re.sub(
            r"'?([A-Za-z_][A-Za-z0-9_]*)'?\[([A-Za-z_][A-Za-z0-9_]*)\]",
            r"\1.\2",
            expr,
        )
        # Convert standalone [col] -> default_table.col
        step2 = re.sub(
            r"(?<![A-Za-z0-9_])\[([A-Za-z_][A-Za-z0-9_]*)\]",
            rf"{default_table}.\1",
            step1,
        )
        return step2

    @staticmethod
    def _split_top_level_args(text: str) -> list[str]:
        parts: list[str] = []
        depth = 0
        current: list[str] = []
        for ch in text:
            if ch == "(":
                depth += 1
                current.append(ch)
            elif ch == ")":
                depth = max(0, depth - 1)
                current.append(ch)
            elif ch == "," and depth == 0:
                parts.append("".join(current).strip())
                current = []
            else:
                current.append(ch)
        if current:
            parts.append("".join(current).strip())
        return parts

    # ------------------------------------------------------------------
    # Build, Validate & Persist Semantic Measure / Calculated Column
    # ------------------------------------------------------------------
    @classmethod
    def create_and_verify_measure(
        cls,
        workspace: OKFWorkspace,
        rag_store: RAGStore,
        connector: "BaseConnector | None",
        name: str,
        table_name: str,
        user_formula: str,
        input_type: str = "dax",  # dax | sql | natural_language
        description: str = "",
        llm_router: LLMRouter | None = None,
    ) -> SemanticMeasure:
        """Compile a DAX/SQL/NL measure into paired DAX + SQL, dry-run verify on live data, and index in RAG."""
        cleaned_formula = user_formula.strip()
        if "=" in cleaned_formula and input_type == "dax":
            lhs, rhs = cleaned_formula.split("=", 1)
            if "(" not in lhs and len(lhs.strip()) < 50:
                cleaned_formula = rhs.strip()

        if input_type == "dax":
            dax_expr = cleaned_formula
            sql_expr = cls.transpile_dax_to_sql(dax_expr, table_name, workspace, llm_router)
        elif input_type == "sql":
            sql_expr = cleaned_formula
            dax_expr = cls.transpile_sql_to_dax(sql_expr, table_name)
        else:
            # Natural Language measure definition
            dax_expr, sql_expr = cls._compile_nl_measure(
                name, table_name, cleaned_formula, workspace, llm_router
            )

        # Detect involved tables referenced in sql_expr
        involved_tables: list[str] = []
        for t in workspace.active_tables():
            if re.search(rf"\b{re.escape(t.name)}\.", sql_expr, re.IGNORECASE) or re.search(
                rf"\b{re.escape(t.name)}\[", dax_expr, re.IGNORECASE
            ):
                involved_tables.append(t.name)
        if not involved_tables and table_name:
            involved_tables = [table_name]

        # Dry-Run Live Validation against the connected engine
        verified = False
        sample_val: Any = None
        if connector is not None:
            try:
                if connector.dialect == "dax":
                    test_dax = f"DEFINE MEASURE '{table_name}'[{name}] = {dax_expr}\nEVALUATE ROW(\"val\", [{name}])"
                    df = connector.execute_query(test_dax, max_rows=1)
                else:
                    if len(involved_tables) == 1:
                        test_sql = f'SELECT ({sql_expr}) AS val FROM "{involved_tables[0]}"'
                    else:
                        # Multi-table measure dry run (e.g. returns / orders): evaluate each table's aggregate in CTE
                        test_sql = cls._build_multitable_measure_test_sql(sql_expr, involved_tables)
                    df = connector.execute_query(test_sql, max_rows=1)
                if not df.empty:
                    raw_v = df.iloc[0, 0]
                    sample_val = round(float(raw_v), 4) if raw_v is not None else 0.0
                    verified = True
            except Exception:
                verified = False

        meas_id = f"meas_{re.sub(r'[^a-z0-9]+', '_', name.lower()).strip('_')}"
        measure = SemanticMeasure(
            id=meas_id,
            name=name,
            table_name=table_name,
            description=description or f"Semantic measure `{name}`",
            user_input_type=input_type,
            dax_expression=dax_expr,
            sql_expression=sql_expr,
            involved_tables=involved_tables,
            verified=verified,
            sample_value=sample_val,
        )

        # Save in OKFWorkspace
        workspace.measures = [m for m in workspace.measures if m.id != meas_id] + [measure]

        # Index into RAG `few_shot_queries` so AI learns user's DAX & SQL measure patterns!
        rag_store.upsert(
            collection="few_shot_queries",
            doc_id=meas_id,
            content=(
                f"Measure Name: [{name}]\n"
                f"Description: {measure.description}\n"
                f"DAX Formula: {name} = {dax_expr}\n"
                f"SQL Formula: {sql_expr}"
            ),
            metadata={
                "type": "measure",
                "name": name,
                "table_name": table_name,
                "dax_expression": dax_expr,
                "sql_expression": sql_expr,
            },
        )
        return measure

    @staticmethod
    def _build_multitable_measure_test_sql(sql_expr: str, involved_tables: list[str]) -> str:
        # Replace SUM(tbl.col) with scalar subqueries for safe multi-fact dry-run validation
        rewritten = re.sub(
            r"(SUM|AVG|MIN|MAX|COUNT)\(\s*([A-Za-z0-9_]+)\.([A-Za-z0-9_]+)\s*\)",
            r"(SELECT \1(\3) FROM \2)",
            sql_expr,
            flags=re.IGNORECASE,
        )
        return f"SELECT ({rewritten}) AS val"

    @classmethod
    def _compile_nl_measure(
        cls,
        name: str,
        table_name: str,
        nl_prompt: str,
        workspace: OKFWorkspace,
        llm_router: LLMRouter | None,
    ) -> tuple[str, str]:
        if llm_router is not None and llm_router.is_configured():
            schema_summary = "\n".join(
                f"- {t.name}: {', '.join(c.name for c in t.columns if not c.is_blocked)}"
                for t in workspace.active_tables()
            )
            try:
                res = llm_router.complete_json(
                    purpose="nl_to_measure",
                    system_prompt=(
                        "You are an expert Power BI DAX and SQL architect. Given the schema and the user's "
                        "natural language metric description, return JSON with keys "
                        "`dax_expression` (valid DAX measure formula) and `sql_expression` (valid SQL aggregate)."
                    ),
                    user_prompt=f"Schema:\n{schema_summary}\nMeasure Name: {name}\nPrimary Table: {table_name}\nRequest: {nl_prompt}",
                )
                if res.get("dax_expression") and res.get("sql_expression"):
                    return str(res["dax_expression"]), str(res["sql_expression"])
            except Exception:
                pass

        # Deterministic heuristic fallback when offline
        tbl = workspace.get_table(table_name) or (workspace.active_tables()[0] if workspace.active_tables() else None)
        num_col = "rev_net_usd"
        if tbl:
            for c in tbl.columns:
                if any(k in c.name.lower() for k in ("rev", "sales", "profit", "amt", "price", "qty")):
                    num_col = c.name
                    break
        t_name = tbl.name if tbl else table_name
        return f"SUM({t_name}[{num_col}])", f"SUM({t_name}.{num_col})"

    @classmethod
    def create_and_verify_calculated_column(
        cls,
        workspace: OKFWorkspace,
        rag_store: RAGStore,
        connector: "BaseConnector | None",
        table_name: str,
        column_name: str,
        user_formula: str,
        input_type: str = "dax",
        description: str = "",
        llm_router: LLMRouter | None = None,
    ) -> CalculatedColumn:
        """Create and verify a row-level Calculated Column (like Power BI Calculated Columns)."""
        cleaned = user_formula.strip()
        if "=" in cleaned and input_type == "dax" and not cleaned.upper().startswith("IF("):
            lhs, rhs = cleaned.split("=", 1)
            if "(" not in lhs and len(lhs.strip()) < 40:
                cleaned = rhs.strip()

        if input_type == "dax":
            dax_expr = cleaned
            sql_expr = cls.transpile_dax_to_sql(dax_expr, table_name, workspace, llm_router)
        else:
            sql_expr = cleaned
            dax_expr = cls.transpile_sql_to_dax(sql_expr, table_name)

        preview: list[Any] = []
        verified = False
        if connector is not None and connector.dialect != "dax":
            try:
                df = connector.execute_query(
                    f'SELECT ({sql_expr}) AS "{column_name}" FROM "{table_name}" LIMIT 5',
                    max_rows=5,
                )
                if not df.empty:
                    preview = df[column_name].tolist()
                    verified = True
            except Exception:
                verified = False

        col_id = f"calc_{table_name}_{re.sub(r'[^a-z0-9]+', '_', column_name.lower()).strip('_')}"
        calc_col = CalculatedColumn(
            id=col_id,
            table_name=table_name,
            name=column_name,
            description=description or f"Calculated column `{column_name}` on `{table_name}`",
            dax_expression=dax_expr,
            sql_expression=sql_expr,
            verified=verified,
            sample_preview=preview,
        )
        workspace.calculated_columns = [
            c for c in workspace.calculated_columns if c.id != col_id
        ] + [calc_col]

        rag_store.upsert(
            collection="few_shot_queries",
            doc_id=col_id,
            content=(
                f"Calculated Column: {table_name}[{column_name}]\n"
                f"DAX Formula: {column_name} = {dax_expr}\n"
                f"SQL Formula: {sql_expr}"
            ),
            metadata={"type": "calculated_column", "table": table_name, "name": column_name},
        )
        return calc_col

    @staticmethod
    def format_dax_define_block(measures: list[SemanticMeasure]) -> str:
        """Generate a Power BI `DEFINE MEASURE` header block for query-scoped DAX execution."""
        if not measures:
            return ""
        lines = ["DEFINE"]
        for m in measures:
            lines.append(f"  MEASURE '{m.table_name}'[{m.name}] = {m.dax_expression}")
        return "\n".join(lines)
