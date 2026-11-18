"""Unit tests for SQLConnector, DuckDBConnector, FederatedConnector, and StatisticalProfiler."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from src.connectors.duckdb_connector import DuckDBConnector
from src.connectors.federated_connector import FederatedConnector
from src.connectors.profiler import StatisticalProfiler
from src.connectors.sql_connector import SQLConnector
from src.knowledge.okf_models import CalculatedColumn


def _make_sqlite_db(db_path: Path) -> str:
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute("CREATE TABLE dept (id INTEGER PRIMARY KEY, name TEXT)")
    cur.execute(
        "CREATE TABLE emp (id INTEGER PRIMARY KEY, dept_id INTEGER, salary REAL, "
        "FOREIGN KEY(dept_id) REFERENCES dept(id))"
    )
    cur.execute("CREATE VIEW emp_summary AS SELECT dept_id, SUM(salary) AS total_sal FROM emp GROUP BY dept_id")
    cur.executemany("INSERT INTO dept VALUES (?, ?)", [(1, "Eng"), (2, "Sales")])
    cur.executemany("INSERT INTO emp VALUES (?, ?, ?)", [(10, 1, 100.0), (11, 1, 150.0), (12, 2, 90.0)])
    conn.commit()
    conn.close()
    return f"sqlite:///{db_path.as_posix()}"


def test_sql_connector_schema_views_and_limit_pushdown(tmp_path: Path) -> None:
    url = _make_sqlite_db(tmp_path / "company.db")
    sql_conn = SQLConnector(url)
    tables, rels, _ = sql_conn.extract_schema()
    names = {t.name for t in tables}
    assert {"dept", "emp"}.issubset(names)
    assert any(r.source_table == "emp" and r.target_table == "dept" for r in rels)

    df = sql_conn.execute_query("SELECT id, salary FROM emp ORDER BY id", max_rows=2)
    assert len(df) == 2

    overlap = sql_conn.compute_column_overlap("emp", "dept_id", "dept", "id")
    assert overlap == 1.0
    sql_conn.close()


def test_duckdb_connector_sanitization_and_calc_column(tmp_path: Path) -> None:
    duck = DuckDBConnector(tmp_path / "ws")
    csv_path = tmp_path / "messy_tbl.csv"
    pd.DataFrame({"1st Col": [10, 20], "Price ($)": [2.5, 4.0]}).to_csv(csv_path, index=False)
    created = duck.ingest_file(str(csv_path))
    assert created == ["messy_tbl"]
    tables, _, _ = duck.extract_schema()
    tbl = next(t for t in tables if t.name == "messy_tbl")
    col_names = [c.name for c in tbl.columns]
    assert len(col_names) == 2

    calc_col = CalculatedColumn(
        id="cc_1",
        table_name="messy_tbl",
        name="double_val",
        sql_expression=f'"{col_names[0]}" * 2',
    )
    preview = duck.apply_calculated_column(calc_col)
    assert sorted(preview) == [20, 40]
    duck.close()


def test_federated_connector_live_routing_and_cleanup(tmp_path: Path) -> None:
    url = _make_sqlite_db(tmp_path / "fed.db")
    fed = FederatedConnector(tmp_path / "ws")
    fed.register_sql_connector("hr_db", url)

    synced = fed.sync_sql_source_to_duckdb("hr_db", ["dept", "emp"])
    assert "hr_db_dept" in synced and "hr_db_emp" in synced

    mirrored_df = fed.execute_query('SELECT name FROM "hr_db"."dept" ORDER BY id')
    assert mirrored_df["name"].tolist() == ["Eng", "Sales"]

    count_df = fed.execute_query("SELECT COUNT(*) AS cnt FROM emp")
    assert int(count_df.iloc[0]["cnt"]) == 3

    fed.remove_source("hr_db", ["hr_db_dept", "hr_db_emp", "dept", "emp"])
    assert "hr_db" not in fed.sql_connectors
    fed.close()


def test_profiler_detects_negative_outliers(tmp_path: Path) -> None:
    duck = DuckDBConnector(tmp_path / "prof_ws")
    csv_path = tmp_path / "neg_tbl.csv"
    pd.DataFrame({"val": [10.0, 20.0, 30.0, 25.0, -999.0]}).to_csv(csv_path, index=False)
    duck.ingest_file(str(csv_path))
    tables, _, _ = duck.extract_schema()
    tbl = next(t for t in tables if t.name == "neg_tbl")
    profiled = StatisticalProfiler.profile_tables(duck, [tbl])
    val_col = profiled[0].get_column("val")
    assert val_col is not None
    assert float(val_col.min_value) == -999.0
    duck.close()
