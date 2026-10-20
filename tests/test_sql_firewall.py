"""Tests for PrivacyFirewall read-only SQL AST enforcement and attack prevention."""

import pytest

from src.security.privacy_firewall import (
    PrivacyFirewall,
    PrivacyMode,
    SecurityViolationError,
)


@pytest.fixture
def firewall():
    return PrivacyFirewall(PrivacyMode.STRICT_SCHEMA_ONLY)


@pytest.mark.parametrize(
    "malicious_sql",
    [
        "DROP TABLE customers",
        "TRUNCATE TABLE orders",
        "ALTER TABLE orders ADD COLUMN hack TEXT",
        "DELETE FROM customers WHERE 1=1",
        "UPDATE customers SET region = 'HACKED'",
        "INSERT INTO customers (id) VALUES ('evil')",
        "COPY customers TO '/tmp/exfil.csv'",
        "ATTACH 'remote.db' AS remote",
        "DETACH DATABASE remote",
        "SELECT * FROM read_csv('/etc/passwd')",
        "SELECT * FROM read_parquet('s3://evil-bucket/keys.parquet')",
        "SELECT * FROM glob('../*')",
        # Multi-statement injection attempt
        "SELECT 1; DROP TABLE orders;",
        # CTE-wrapped modification attempt
        "WITH hacked AS (DELETE FROM orders RETURNING *) SELECT * FROM hacked",
    ],
)
def test_sqlglot_firewall_blocks_mutations_and_file_reads(firewall, malicious_sql):
    with pytest.raises(SecurityViolationError):
        firewall.validate_sql_read_only(malicious_sql, dialect="duckdb")


def test_sqlglot_firewall_allows_complex_analytical_select(firewall):
    safe_query = """
    WITH regional_rev AS (
        SELECT c.region, SUM(o.rev_net_usd) AS total_rev
        FROM orders o
        JOIN customers c ON o.customer_id = c.customer_id
        WHERE o.ord_stat <> 'X'
        GROUP BY c.region
    )
    SELECT region, total_rev, RANK() OVER (ORDER BY total_rev DESC) AS rank
    FROM regional_rev
    LIMIT 100
    """
    validated = firewall.validate_sql_read_only(safe_query, dialect="duckdb")
    assert "regional_rev" in validated


def test_sqlglot_firewall_blocked_column_enforcement(firewall):
    query = "SELECT customer_id, contact_email, region FROM customers"
    with pytest.raises(SecurityViolationError):
        firewall.validate_sql_read_only(
            query,
            dialect="duckdb",
            blocked_columns={"contact_email"},
        )
