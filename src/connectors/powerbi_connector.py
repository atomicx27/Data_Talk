"""Unified Power BI Connector for Data_Talk.

Supports all 3 Power BI Connection Modes requested by the user:
1. **Offline `.pbix` File Mode (`powerbi_pbix`)**:
   - Uses `pbixray` (if installed) OR built-in `.pbix` ZIP `DataModelSchema` / JSON metadata unpacker
     to extract tables, columns, VertiPaq relationships, and existing DAX measures into DuckDB + OKF.
2. **Live Power BI Desktop Local Mode (`powerbi_desktop`)**:
   - Scans Windows `%LOCALAPPDATA%\\Microsoft\\PowerBI Desktop\\AnalysisServicesWorkspaces\\*\\msmdsrv.port.txt`
     to auto-discover running local Analysis Services Tabular ports (`localhost:<port>`).
   - Queries `$SYSTEM.TMSCHEMA_TABLES`, `COLUMNS`, `RELATIONSHIPS`, `MEASURES` and runs live `DEFINE MEASURE ... EVALUATE` DAX queries.
3. **Power BI Cloud Service REST API Mode (`powerbi_cloud`)**:
   - Connects via Azure AD Bearer Token (or Service Principal `tenant_id` / `client_id` / `client_secret`)
     to `https://api.powerbi.com/v1.0/myorg/datasets/{dataset_id}/executeQueries`.
   - Executes live DAX (`DEFINE MEASURE ... EVALUATE SUMMARIZECOLUMNS(...)`) and `INFO.TABLES()` / `INFO.MEASURES()` queries.
"""

from __future__ import annotations

import json
import os
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from src.connectors.base import BaseConnector
from src.connectors.duckdb_connector import DuckDBConnector
from src.knowledge.okf_models import (
    ColumnProfile,
    JoinCardinality,
    JoinStatus,
    RelationshipEdge,
    SemanticMeasure,
    TableSchema,
)
from src.security.privacy_firewall import PrivacyFirewall


class PowerBIConnector(BaseConnector):
    """Unified connector for Power BI (.pbix files, live Power BI Desktop localhost, and Power BI Cloud REST API)."""

    def __init__(
        self,
        mode: str = "powerbi_cloud",
        workspace_dir: str | Path = "workspaces/default",
        pbix_path: str | Path | None = None,
        localhost_port: int | None = None,
        dataset_id: str | None = None,
        access_token: str | None = None,
        tenant_id: str | None = None,
        client_id: str | None = None,
        client_secret: str | None = None,
    ) -> None:
        self.mode = mode  # powerbi_pbix | powerbi_desktop | powerbi_cloud
        self.source_type = mode
        self.dialect = "duckdb" if mode == "powerbi_pbix" else "dax"
        self.workspace_dir = Path(workspace_dir)
        self.pbix_path = Path(pbix_path) if pbix_path else None
        self.localhost_port = localhost_port
        self.dataset_id = dataset_id
        self._access_token = access_token
        self._tenant_id = tenant_id
        self._client_id = client_id
        self._client_secret = client_secret
        self._duckdb: DuckDBConnector | None = (
            DuckDBConnector(self.workspace_dir) if mode == "powerbi_pbix" else None
        )

    # ------------------------------------------------------------------
    # Windows Power BI Desktop Local Instance Discovery
    # ------------------------------------------------------------------
    @staticmethod
    def discover_local_desktop_instances() -> list[dict[str, Any]]:
        """Auto-discover running Power BI Desktop Analysis Services instances on Windows."""
        instances: list[dict[str, Any]] = []
        local_app_data = os.environ.get("LOCALAPPDATA")
        if not local_app_data:
            return instances

        candidates = [
            Path(local_app_data) / "Microsoft" / "PowerBI Desktop" / "AnalysisServicesWorkspaces",
            Path(local_app_data)
            / "Packages"
            / "Microsoft.MicrosoftPowerBIDesktop_8wekyb3d8bbwe"
            / "LocalState"
            / "AnalysisServicesWorkspaces",
        ]
        for base in candidates:
            if not base.exists():
                continue
            for ws_folder in base.iterdir():
                port_file = ws_folder / "Data" / "msmdsrv.port.txt"
                if not port_file.exists():
                    port_file = ws_folder / "msmdsrv.port.txt"
                if port_file.exists():
                    try:
                        raw_bytes = port_file.read_bytes()
                        port_text = (
                            raw_bytes.decode("utf-16-le", errors="ignore").strip()
                            or raw_bytes.decode("utf-8", errors="ignore").strip()
                        )
                        port_digits = "".join(ch for ch in port_text if ch.isdigit())
                        if port_digits:
                            instances.append(
                                {
                                    "workspace_folder": ws_folder.name,
                                    "port": int(port_digits),
                                    "connection_string": f"Provider=MSOLAP;Data Source=localhost:{int(port_digits)};",
                                }
                            )
                    except Exception:
                        continue
        return instances

    # ------------------------------------------------------------------
    # Azure AD OAuth Token Helper for Power BI Cloud
    # ------------------------------------------------------------------
    def _get_cloud_bearer_token(self) -> str:
        if self._access_token:
            return self._access_token
        if not (self._tenant_id and self._client_id and self._client_secret):
            raise ValueError(
                "Power BI Cloud requires either an Access Token or (Tenant ID, Client ID, Client Secret)."
            )
        token_url = f"https://login.microsoftonline.com/{self._tenant_id}/oauth2/v2.0/token"
        resp = requests.post(
            token_url,
            data={
                "grant_type": "client_credentials",
                "client_id": self._client_id,
                "client_secret": self._client_secret,
                "scope": "https://analysis.windows.net/powerbi/api/.default",
            },
            timeout=15,
        )
        resp.raise_for_status()
        token = resp.json().get("access_token")
        if not token:
            raise ValueError("Azure AD token response did not include an access_token.")
        self._access_token = token
        return token

    # ------------------------------------------------------------------
    # Schema, Relationship & Existing DAX Measure Extraction
    # ------------------------------------------------------------------
    def extract_schema(
        self,
        blocked_tables: set[str] | None = None,
        blocked_columns: set[str] | None = None,
    ) -> tuple[list[TableSchema], list[RelationshipEdge], list[SemanticMeasure]]:
        if self.mode == "powerbi_pbix":
            return self._extract_pbix_schema(blocked_tables, blocked_columns)
        if self.mode == "powerbi_desktop":
            return self._extract_desktop_schema(blocked_tables, blocked_columns)
        return self._extract_cloud_schema(blocked_tables, blocked_columns)

    def _extract_pbix_schema(
        self,
        blocked_tables: set[str] | None = None,
        blocked_columns: set[str] | None = None,
    ) -> tuple[list[TableSchema], list[RelationshipEdge], list[SemanticMeasure]]:
        if not self.pbix_path or not self.pbix_path.exists():
            raise FileNotFoundError(f".pbix file not found: {self.pbix_path}")
        assert self._duckdb is not None

        relationships: list[RelationshipEdge] = []
        measures: list[SemanticMeasure] = []

        # Attempt pbixray first for binary VertiPaq DataModel extraction
        try:
            from pbixray import PBIXRay  # type: ignore

            model = PBIXRay(str(self.pbix_path))
            for t_name in model.tables:
                df = model.get_table(t_name)
                self._duckdb.ingest_dataframe(str(t_name), df)

            if hasattr(model, "dax_measures") and model.dax_measures is not None:
                m_df = model.dax_measures
                for idx, mrow in m_df.iterrows():
                    m_name = str(mrow.get("Name") or f"Measure_{idx}")
                    m_expr = str(mrow.get("Expression") or "")
                    m_tbl = str(mrow.get("TableName") or "Metrics")
                    if m_expr:
                        measures.append(
                            SemanticMeasure(
                                id=f"pbi_meas_{idx}_{m_name.lower().replace(' ', '_')}",
                                name=m_name,
                                table_name=m_tbl,
                                description=str(mrow.get("Description") or "Imported from .pbix model"),
                                user_input_type="dax",
                                dax_expression=m_expr,
                                sql_expression="",
                                verified=True,
                            )
                        )

            if hasattr(model, "relationships") and model.relationships is not None:
                r_df = model.relationships
                for idx, rrow in r_df.iterrows():
                    src_t = str(rrow.get("FromTableName") or "")
                    src_c = str(rrow.get("FromColumnName") or "")
                    tgt_t = str(rrow.get("ToTableName") or "")
                    tgt_c = str(rrow.get("ToColumnName") or "")
                    if src_t and tgt_t:
                        relationships.append(
                            RelationshipEdge(
                                id=f"pbix_rel_{idx}",
                                source_table=src_t,
                                source_column=src_c,
                                target_table=tgt_t,
                                target_column=tgt_c,
                                cardinality=JoinCardinality.MANY_TO_ONE,
                                confidence=1.0,
                                discovery_method="powerbi_dmv",
                                status=JoinStatus.APPROVED,
                                overlap_ratio=1.0,
                            )
                        )
        except ImportError:
            # Fallback: inspect DataModelSchema JSON inside .pbit / .pbix zip archive if present
            self._extract_pbix_zip_metadata(relationships, measures)

        tables, _, _ = self._duckdb.extract_schema(blocked_tables, blocked_columns)
        return tables, relationships, measures

    def _extract_pbix_zip_metadata(
        self,
        relationships: list[RelationshipEdge],
        measures: list[SemanticMeasure],
    ) -> None:
        if not self.pbix_path or not zipfile.is_zipfile(self.pbix_path):
            return
        with zipfile.ZipFile(self.pbix_path, "r") as zf:
            if "DataModelSchema" in zf.namelist():
                raw = zf.read("DataModelSchema")
                text = raw.decode("utf-16-le", errors="ignore") or raw.decode("utf-8", errors="ignore")
                data = json.loads(text)
                model_obj = data.get("model", {})
                for tbl in model_obj.get("tables", []):
                    t_name = tbl.get("name", "Table")
                    for idx, m in enumerate(tbl.get("measures", [])):
                        expr = m.get("expression", "")
                        if isinstance(expr, list):
                            expr = "\n".join(expr)
                        measures.append(
                            SemanticMeasure(
                                id=f"pbit_meas_{t_name}_{idx}",
                                name=m.get("name", f"Measure_{idx}"),
                                table_name=t_name,
                                dax_expression=str(expr),
                                sql_expression="",
                                verified=True,
                            )
                        )
                for idx, rel in enumerate(model_obj.get("relationships", [])):
                    relationships.append(
                        RelationshipEdge(
                            id=f"pbit_rel_{idx}",
                            source_table=rel.get("fromTable", ""),
                            source_column=rel.get("fromColumn", ""),
                            target_table=rel.get("toTable", ""),
                            target_column=rel.get("toColumn", ""),
                            cardinality=JoinCardinality.MANY_TO_ONE,
                            confidence=1.0,
                            discovery_method="powerbi_dmv",
                            status=JoinStatus.APPROVED,
                        )
                    )

    def _extract_desktop_schema(
        self,
        blocked_tables: set[str] | None = None,
        blocked_columns: set[str] | None = None,
    ) -> tuple[list[TableSchema], list[RelationshipEdge], list[SemanticMeasure]]:
        tables_df = self.execute_query("EVALUATE INFO.TABLES()", max_rows=500)
        cols_df = self.execute_query("EVALUATE INFO.COLUMNS()", max_rows=5000)
        meas_df = self.execute_query("EVALUATE INFO.MEASURES()", max_rows=2000)
        return self._build_schema_from_dax_info(
            tables_df, cols_df, meas_df, blocked_tables, blocked_columns
        )

    def _extract_cloud_schema(
        self,
        blocked_tables: set[str] | None = None,
        blocked_columns: set[str] | None = None,
    ) -> tuple[list[TableSchema], list[RelationshipEdge], list[SemanticMeasure]]:
        tables_df = self.execute_query("EVALUATE INFO.TABLES()", max_rows=500)
        cols_df = self.execute_query("EVALUATE INFO.COLUMNS()", max_rows=5000)
        meas_df = self.execute_query("EVALUATE INFO.MEASURES()", max_rows=2000)
        return self._build_schema_from_dax_info(
            tables_df, cols_df, meas_df, blocked_tables, blocked_columns
        )

    @staticmethod
    def _build_schema_from_dax_info(
        tables_df: pd.DataFrame,
        cols_df: pd.DataFrame,
        meas_df: pd.DataFrame,
        blocked_tables: set[str] | None,
        blocked_columns: set[str] | None,
    ) -> tuple[list[TableSchema], list[RelationshipEdge], list[SemanticMeasure]]:
        blocked_t = {t.lower() for t in (blocked_tables or set())}
        blocked_c = {c.lower() for c in (blocked_columns or set())}
        table_id_map: dict[Any, TableSchema] = {}

        for _, row in tables_df.iterrows():
            t_id = row.get("[ID]") or row.get("ID")
            t_name = str(row.get("[Name]") or row.get("Name") or f"Table_{t_id}")
            if t_name.startswith(("DateTableTemplate_", "LocalDateTable_")):
                continue
            table_id_map[t_id] = TableSchema(
                name=t_name,
                description=str(row.get("[Description]") or row.get("Description") or ""),
                is_blocked=t_name.lower() in blocked_t,
            )

        for _, row in cols_df.iterrows():
            t_id = row.get("[TableID]") or row.get("TableID")
            tbl = table_id_map.get(t_id)
            if not tbl:
                continue
            c_name = str(
                row.get("[ExplicitName]")
                or row.get("ExplicitName")
                or row.get("[InferredName]")
                or row.get("InferredName")
                or ""
            )
            if not c_name or c_name.startswith("RowNumber"):
                continue
            tbl.columns.append(
                ColumnProfile(
                    name=c_name,
                    data_type=str(row.get("[DataType]") or row.get("DataType") or "STRING"),
                    is_pii=PrivacyFirewall.is_pii_column_name(c_name),
                    is_blocked=c_name.lower() in blocked_c,
                )
            )

        measures: list[SemanticMeasure] = []
        for idx, row in meas_df.iterrows():
            m_name = str(row.get("[Name]") or row.get("Name") or f"Measure_{idx}")
            m_expr = str(row.get("[Expression]") or row.get("Expression") or "")
            t_id = row.get("[TableID]") or row.get("TableID")
            tbl = table_id_map.get(t_id)
            if m_expr:
                measures.append(
                    SemanticMeasure(
                        id=f"pbi_info_meas_{idx}",
                        name=m_name,
                        table_name=tbl.name if tbl else "Metrics",
                        description=str(row.get("[Description]") or row.get("Description") or ""),
                        dax_expression=m_expr,
                        sql_expression="",
                        verified=True,
                    )
                )

        return list(table_id_map.values()), [], measures

    # ------------------------------------------------------------------
    # Query Execution (DuckDB for offline .pbix, DAX for Desktop & Cloud)
    # ------------------------------------------------------------------
    def execute_query(
        self,
        query: str,
        max_rows: int = 10000,
        blocked_tables: set[str] | None = None,
        blocked_columns: set[str] | None = None,
    ) -> pd.DataFrame:
        if self.mode == "powerbi_pbix":
            assert self._duckdb is not None
            return self._duckdb.execute_query(query, max_rows, blocked_tables, blocked_columns)

        validated_dax = PrivacyFirewall.validate_dax_read_only(query)
        if self.mode == "powerbi_desktop":
            return self._execute_desktop_dax(validated_dax, max_rows)
        return self._execute_cloud_dax(validated_dax, max_rows)

    def _execute_desktop_dax(self, dax_query: str, max_rows: int) -> pd.DataFrame:
        if not self.localhost_port:
            instances = self.discover_local_desktop_instances()
            if not instances:
                raise RuntimeError(
                    "No running Power BI Desktop instance found on localhost. "
                    "Open your .pbix in Power BI Desktop or use Offline .pbix Upload mode."
                )
            self.localhost_port = int(instances[0]["port"])

        try:
            from pyadomd import Pyadomd  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "Live Power BI Desktop connection requires 'pyadomd' and Windows MSOLAP driver. "
                "Alternatively, upload your .pbix file directly in Offline .pbix Mode."
            ) from exc

        conn_str = f"Provider=MSOLAP;Data Source=localhost:{self.localhost_port};"
        with Pyadomd(conn_str) as conn:
            with conn.cursor().execute(dax_query) as cur:
                cols = [col.name for col in cur.description]
                rows = cur.fetchmany(max_rows)
                return pd.DataFrame(rows, columns=cols)

    def _execute_cloud_dax(self, dax_query: str, max_rows: int) -> pd.DataFrame:
        if not self.dataset_id:
            raise ValueError("Power BI Cloud requires a valid dataset_id.")
        token = self._get_cloud_bearer_token()
        url = f"https://api.powerbi.com/v1.0/myorg/datasets/{self.dataset_id}/executeQueries"
        resp = requests.post(
            url,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
            json={
                "queries": [{"query": dax_query}],
                "serializerSettings": {"includeNulls": True},
            },
            timeout=30,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"Power BI executeQueries error ({resp.status_code}): {resp.text}")
        payload = resp.json()
        rows = (
            payload.get("results", [{}])[0]
            .get("tables", [{}])[0]
            .get("rows", [])
        )
        return pd.DataFrame(rows[:max_rows])

    def compute_column_overlap(
        self,
        src_table: str,
        src_col: str,
        tgt_table: str,
        tgt_col: str,
    ) -> float:
        if self.mode == "powerbi_pbix" and self._duckdb is not None:
            return self._duckdb.compute_column_overlap(src_table, src_col, tgt_table, tgt_col)
        return 0.0
