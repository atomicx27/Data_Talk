"""Tests for bidirectional DAX <-> SQL Measure Compiler."""

from src.agents.measure_compiler import MeasureCompiler
from src.knowledge.okf_models import SemanticMeasure


def test_dax_to_sql_transpilation_basic():
    # 1. Simple SUM
    sql1 = MeasureCompiler.transpile_dax_to_sql("SUM('orders'[rev_net_usd])", default_table="orders")
    assert "SUM" in sql1
    assert "orders.rev_net_usd" in sql1

    # 2. DISTINCTCOUNT
    sql2 = MeasureCompiler.transpile_dax_to_sql("DISTINCTCOUNT('orders'[customer_id])", default_table="orders")
    assert "COUNT(DISTINCT orders.customer_id)" in sql2

    # 3. DIVIDE with safe zero division
    sql3 = MeasureCompiler.transpile_dax_to_sql(
        "DIVIDE(SUM('orders'[profit]), SUM('orders'[rev_net_usd]))",
        default_table="orders",
    )
    assert "NULLIF" in sql3

    # 4. CALCULATE with filter condition
    sql4 = MeasureCompiler.transpile_dax_to_sql(
        "CALCULATE(SUM('orders'[rev_net_usd]), 'orders'[ord_stat] <> \"X\")",
        default_table="orders",
    )
    assert "CASE WHEN" in sql4
    assert "ord_stat" in sql4


def test_format_dax_define_block():
    measures = [
        SemanticMeasure(
            id="m1",
            name="Net Revenue",
            table_name="orders",
            dax_expression="SUM('orders'[rev_net_usd])",
            sql_expression="SUM(orders.rev_net_usd)",
            description="Total net revenue",
        ),
        SemanticMeasure(
            id="m2",
            name="Total Profit",
            table_name="orders",
            dax_expression="SUM('orders'[rev_net_usd]) - SUM('orders'[cogs_usd])",
            sql_expression="SUM(orders.rev_net_usd - orders.cogs_usd)",
            description="Gross profit",
        ),
    ]

    block = MeasureCompiler.format_dax_define_block(measures)
    assert "DEFINE" in block
    assert "MEASURE 'orders'[Net Revenue]" in block
    assert "MEASURE 'orders'[Total Profit]" in block
