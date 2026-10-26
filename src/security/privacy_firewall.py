"""6-Layer Zero-Data-Leak & Enterprise Security Firewall for Data_Talk.

Responsibilities:
1. Automated PII Detection & Redaction during Data Profiling (Emails, Phones, SSNs/PAN, Credit Cards, IPs, Secrets).
2. Indirect Prompt-Injection Sanitization (<untrusted_data_sample> isolation & control-char stripping).
3. 3-Layer Read-Only SQL AST Firewall (via sqlglot) & DAX Sandbox Guard.
4. Encrypted Local Credential Vault (cryptography.fernet + .env) so secrets never touch okf.yaml.
5. Outbound LLM Payload Inspector & Secret Scrubbing Filter.
"""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import pandas as pd
import sqlglot
from cryptography.fernet import Fernet, InvalidToken
from sqlglot import exp

_LOGGER = logging.getLogger(__name__)


class PrivacyMode(str, Enum):
    """Controls what data is allowed to leave localhost to an external LLM."""

    STRICT_SCHEMA_ONLY = "strict_schema_only"  # Default: Zero rows or result stats sent to Cloud LLM
    ANONYMIZED_SUMMARY = "anonymized_summary"  # Only aggregated non-PII summary stats (<=20 rows) sent
    AIR_GAPPED_LOCAL = "air_gapped_local"      # 100% local Ollama + local SQLite/NumPy RAG (0 bytes leave localhost)


class SecurityViolationError(Exception):
    """Raised when a query or payload violates the read-only or privacy firewall."""


# ---------------------------------------------------------------------------
# 1. Automated PII Detection & Sample Sanitization
# ---------------------------------------------------------------------------

PII_COLUMN_NAME_PATTERNS = re.compile(
    r"(^|_)(ssn|social_security|pan_num|aadhaar|tax_id|passport|credit_card|cc_num|cvv|"
    r"password|passwd|pwd|secret|token|api_key|private_key|email|e_mail|phone|mobile|cell|"
    r"dob|date_of_birth|birth_date|salary|compensation|bank_account|iban|routing_number)($|_)",
    re.IGNORECASE,
)

PII_VALUE_REGEXES: list[tuple[str, re.Pattern[str]]] = [
    ("EMAIL_PII", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")),
    ("SSN_PII", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("CREDIT_CARD_PII", re.compile(r"\b(?:\d[ -]*?){13,16}\b")),
    ("PHONE_PII", re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")),
    ("IPV4_PII", re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")),
    (
        "SECRET_KEY_PII",
        re.compile(
            r"\b(?:sk-[A-Za-z0-9_-]{16,}|nvapi-[A-Za-z0-9_-]{16,}|gsk_[A-Za-z0-9_-]{16,}|Bearer\s+[A-Za-z0-9._\-]+)\b"
        ),
    ),
]

PROMPT_INJECTION_PATTERNS = re.compile(
    r"(ignore\s+(all\s+)?(previous|prior)\s+instructions|system\s*prompt|"
    r"you\s+are\s+now|disregard\s+above|drop\s+table|delete\s+from|xp_cmdshell)",
    re.IGNORECASE,
)

FORBIDDEN_SQL_FUNCTIONS = {
    "pg_sleep",
    "sleep",
    "waitfor",
    "benchmark",
    "xp_cmdshell",
    "lo_import",
    "lo_export",
    "pg_read_file",
    "pg_read_binary_file",
    "pg_ls_dir",
    "read_csv",
    "read_csv_auto",
    "read_parquet",
    "read_json",
    "write_parquet",
    "glob",
}

FORBIDDEN_SYSTEM_SCHEMAS = {
    "pg_catalog",
    "information_schema",
    "sys",
    "mysql",
    "performance_schema",
    "msdb",
    "master",
}


class PrivacyFirewall:
    """Centralized privacy and query security guard."""

    def __init__(self, privacy_mode: PrivacyMode = PrivacyMode.STRICT_SCHEMA_ONLY) -> None:
        self.privacy_mode = privacy_mode
        self.audit_log: list[dict[str, Any]] = []

    @staticmethod
    def is_pii_column_name(column_name: str) -> bool:
        """Return True if the column name matches known PII or credential patterns."""
        return bool(PII_COLUMN_NAME_PATTERNS.search(column_name.strip()))

    @classmethod
    def redact_text(cls, text: str) -> str:
        """Scrub emails, SSNs, credit cards, phone numbers, and API keys from text."""
        if not text:
            return text
        scrubbed = str(text)
        for label, pattern in PII_VALUE_REGEXES:
            scrubbed = pattern.sub(f"<REDACTED_{label}>", scrubbed)
        # Scrub database connection strings with embedded passwords
        scrubbed = re.sub(
            r"://([^:/\s]+):([^@/\s]+)@",
            r"://\1:<REDACTED_PASSWORD>@",
            scrubbed,
        )
        return scrubbed

    @classmethod
    def sanitize_profile_samples(
        cls,
        column_name: str,
        raw_samples: list[Any],
        distinct_count: int,
        max_cardinality_for_sampling: int = 100,
    ) -> tuple[list[str], bool]:
        """Sanitize column sample values for safe storage in OKF and prompts.

        Returns:
            (sanitized_samples, is_pii_detected)
        """
        if cls.is_pii_column_name(column_name):
            return ["<REDACTED_PII_COLUMN>"], True

        # High-cardinality columns (IDs, names, free-text comments) are never sampled
        if distinct_count > max_cardinality_for_sampling:
            return [], False

        sanitized: list[str] = []
        pii_found = False
        for val in raw_samples[:65]:
            if val is None:
                continue
            s_val = str(val).strip()
            # Strip control characters & newlines
            s_val = re.sub(r"[\r\n\t\x00-\x1f]+", " ", s_val)[:80]
            # Check for prompt injection inside database values
            if PROMPT_INJECTION_PATTERNS.search(s_val):
                sanitized.append("<REDACTED_UNTRUSTED_TEXT>")
                continue
            redacted_val = cls.redact_text(s_val)
            if redacted_val != s_val:
                pii_found = True
            sanitized.append(redacted_val)

        return sanitized, pii_found

    @staticmethod
    def wrap_untrusted_samples(samples: list[str]) -> str:
        """Wrap database sample values in <untrusted_data_sample> tags to prevent indirect prompt injection."""
        if not samples:
            return "[]"
        safe_json = json.dumps(samples, ensure_ascii=True)
        return f"<untrusted_data_sample>{safe_json}</untrusted_data_sample>"

    @classmethod
    def validate_sql_read_only(
        cls,
        sql: str,
        dialect: str | None = None,
        blocked_tables: set[str] | None = None,
        blocked_columns: set[str] | None = None,
    ) -> str:
        """Validate via sqlglot AST that SQL is strictly a single read-only SELECT/WITH query.

        Raises SecurityViolationError if any mutating statement, system catalog,
        dangerous function, or blocked table/column is referenced.
        """
        cleaned_sql = (sql or "").strip()
        if not cleaned_sql:
            raise SecurityViolationError("Empty SQL query is not permitted.")

        # Quick pre-check for dangerous keywords that some dialects don't parse into standard AST nodes
        upper_tokens = set(re.findall(r"\b[A-Z_]+\b", cleaned_sql.upper()))
        forbidden_keywords = {
            "DROP",
            "DELETE",
            "UPDATE",
            "INSERT",
            "ALTER",
            "TRUNCATE",
            "CREATE",
            "REPLACE",
            "GRANT",
            "REVOKE",
            "COPY",
            "ATTACH",
            "DETACH",
            "INSTALL",
            "LOAD",
            "CALL",
            "EXEC",
            "EXECUTE",
            "MERGE",
            "VACUUM",
        }
        hit_keywords = upper_tokens.intersection(forbidden_keywords)
        if hit_keywords:
            raise SecurityViolationError(
                f"Blocked non-read-only SQL keyword(s): {', '.join(sorted(hit_keywords))}"
            )

        try:
            statements = sqlglot.parse(cleaned_sql, read=dialect)
        except Exception as exc:
            raise SecurityViolationError(f"SQL AST parse error: {exc}") from exc

        non_empty_stmts = [s for s in statements if s is not None]
        if len(non_empty_stmts) != 1:
            raise SecurityViolationError(
                f"Multi-statement SQL queries are forbidden (found {len(non_empty_stmts)} statements)."
            )

        stmt = non_empty_stmts[0]
        if not isinstance(stmt, (exp.Select, exp.Union, exp.Intersect, exp.Except)):
            raise SecurityViolationError(
                f"Only SELECT queries are permitted (got AST node: {type(stmt).__name__})."
            )

        # Inspect tables for system catalogs and user-blocked tables
        blocked_tbl_lower = {t.lower() for t in (blocked_tables or set())}
        for tbl in stmt.find_all(exp.Table):
            db_or_schema = (tbl.db or "").lower()
            tbl_name = (tbl.name or "").lower()
            if db_or_schema in FORBIDDEN_SYSTEM_SCHEMAS or tbl_name.startswith("pg_"):
                raise SecurityViolationError(
                    f"Access to system catalog '{db_or_schema}.{tbl_name}' is forbidden."
                )
            if tbl_name in blocked_tbl_lower:
                raise SecurityViolationError(
                    f"Query references blocked/restricted table '{tbl.name}'."
                )

        # Inspect columns for user-blocked columns
        blocked_col_lower = {c.lower() for c in (blocked_columns or set())}
        if blocked_col_lower:
            for col in stmt.find_all(exp.Column):
                col_name = (col.name or "").lower()
                full_col = f"{(col.table or '').lower()}.{col_name}".strip(".")
                if col_name in blocked_col_lower or full_col in blocked_col_lower:
                    raise SecurityViolationError(
                        f"Query references blocked/PII column '{col.name}'."
                    )

        # Inspect function calls for sleep / file access
        for func in stmt.find_all(exp.Func):
            func_name = func.sql_name().lower()
            raw_name = (func.name or "").lower()
            if func_name in FORBIDDEN_SQL_FUNCTIONS or raw_name in FORBIDDEN_SQL_FUNCTIONS:
                raise SecurityViolationError(
                    f"Forbidden SQL function '{raw_name or func_name}' blocked by AST firewall."
                )

        return cleaned_sql

    @classmethod
    def validate_dax_read_only(cls, dax_query: str) -> str:
        """Validate that a DAX query only performs read-only DEFINE + EVALUATE execution."""
        cleaned = (dax_query or "").strip()
        if not cleaned:
            raise SecurityViolationError("Empty DAX query is not permitted.")

        if re.search(r"<\s*(Alter|Drop|Create|Process|Batch|Delete|Insert|Update)\b", cleaned, re.I):
            raise SecurityViolationError("XMLA administrative commands are forbidden in DAX sandbox.")

        if not re.search(r"\bEVALUATE\b", cleaned, re.IGNORECASE):
            raise SecurityViolationError("DAX query must contain an EVALUATE clause.")

        return cleaned

    def prepare_dataframe_summary_for_llm(self, df: pd.DataFrame) -> dict[str, Any]:
        """Prepare a privacy-preserving summary of a query result DataFrame based on PrivacyMode."""
        if self.privacy_mode in (PrivacyMode.STRICT_SCHEMA_ONLY,):
            return {
                "privacy_mode": self.privacy_mode.value,
                "row_count": len(df),
                "columns": list(df.columns),
                "note": "Raw rows withheld from Cloud LLM by Strict Schema-Only Privacy Firewall.",
            }

        # ANONYMIZED_SUMMARY or AIR_GAPPED_LOCAL: strip any PII columns and cap at 15 aggregate rows
        safe_cols = [c for c in df.columns if not self.is_pii_column_name(str(c))]
        safe_df = df[safe_cols].head(15).copy()
        for col in safe_df.select_dtypes(include=["object", "string"]).columns:
            safe_df[col] = safe_df[col].astype(str).map(self.redact_text)

        return {
            "privacy_mode": self.privacy_mode.value,
            "row_count": len(df),
            "columns": safe_cols,
            "preview_rows": safe_df.to_dict(orient="records"),
        }

    def inspect_and_scrub_outbound_prompt(
        self,
        purpose: str,
        model: str,
        messages: list[dict[str, str]],
    ) -> list[dict[str, str]]:
        """Scrub any accidental credentials/PII from outbound LLM messages and log for UI inspection."""
        if self.privacy_mode == PrivacyMode.AIR_GAPPED_LOCAL:
            if not (
                model.startswith("ollama/")
                or model.startswith("local/")
                or "localhost" in model
            ):
                raise SecurityViolationError(
                    f"Air-Gapped Local Mode is active, but cloud model '{model}' was requested. "
                    "Switch to an 'ollama/...' model or change Privacy Mode."
                )

        scrubbed_messages: list[dict[str, str]] = []
        for msg in messages:
            scrubbed_content = self.redact_text(msg.get("content", ""))
            scrubbed_messages.append(
                {"role": msg.get("role", "user"), "content": scrubbed_content}
            )

        self.audit_log.append(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "purpose": purpose,
                "model": model,
                "privacy_mode": self.privacy_mode.value,
                "messages": scrubbed_messages,
            }
        )
        # Keep last 50 outbound payloads in memory for UI inspection
        if len(self.audit_log) > 50:
            self.audit_log = self.audit_log[-50:]

        return scrubbed_messages


# ---------------------------------------------------------------------------
# 2. Local Encrypted Credential Vault (Never stores secrets in okf.yaml)
# ---------------------------------------------------------------------------


class CredentialVault:
    """Stores database connection strings & tokens in a Fernet-encrypted local file (.secrets/vault.enc).

    Master-key resolution order (most to least secure):
    1. ``DATA_TALK_MASTER_KEY`` environment variable.
    2. OS keyring (Windows Credential Manager / macOS Keychain / Secret Service) if the optional
       ``keyring`` package is installed.
    3. Legacy ``master.key`` file next to the vault (kept for backwards compatibility, logs a warning).
    """

    KEYRING_SERVICE = "data_talk"
    KEYRING_USERNAME = "vault_master_key"

    def __init__(self, base_dir: str | Path = ".secrets", use_keyring: bool | None = None) -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.key_file = self.base_dir / "master.key"
        self.vault_file = self.base_dir / "vault.enc"
        if use_keyring is None:
            use_keyring = os.environ.get("DATA_TALK_USE_KEYRING", "1").lower() not in {"0", "false", "no"}
        self._use_keyring = use_keyring
        self.key_source = "unknown"
        self._fernet = Fernet(self._load_or_create_key())

    @staticmethod
    def _keyring_module() -> Any | None:
        try:
            import keyring  # type: ignore[import-not-found]

            return keyring
        except Exception:
            return None

    def _load_or_create_key(self) -> bytes:
        env_key = os.environ.get("DATA_TALK_MASTER_KEY")
        if env_key:
            self.key_source = "env"
            return env_key.strip().encode("utf-8")

        # Legacy key file takes precedence over creating a new keyring key so existing vaults keep decrypting.
        if self.key_file.exists():
            self.key_source = "file"
            _LOGGER.warning(
                "CredentialVault is using a plaintext master key file (%s). Set DATA_TALK_MASTER_KEY "
                "or install 'keyring' and move the key to the OS credential store for better protection.",
                self.key_file,
            )
            return self.key_file.read_bytes().strip()

        kr = self._keyring_module() if self._use_keyring else None
        if kr is not None:
            try:
                stored = kr.get_password(self.KEYRING_SERVICE, self.KEYRING_USERNAME)
                if stored:
                    self.key_source = "keyring"
                    return stored.encode("utf-8")
                new_key = Fernet.generate_key()
                kr.set_password(self.KEYRING_SERVICE, self.KEYRING_USERNAME, new_key.decode("utf-8"))
                self.key_source = "keyring"
                return new_key
            except Exception as exc:  # keyring backend unavailable (e.g. headless Linux)
                _LOGGER.warning("OS keyring unavailable (%s); falling back to master.key file.", exc)

        key = Fernet.generate_key()
        self.key_file.write_bytes(key)
        try:
            os.chmod(self.key_file, 0o600)
        except OSError:
            pass
        self.key_source = "file"
        return key

    def _read_all(self) -> dict[str, str]:
        if not self.vault_file.exists():
            return {}
        try:
            decrypted = self._fernet.decrypt(self.vault_file.read_bytes())
        except InvalidToken as exc:
            raise VaultDecryptionError(
                f"Cannot decrypt {self.vault_file}: the master key (source: {self.key_source}) does not match. "
                "Restore the original DATA_TALK_MASTER_KEY / master.key, or delete vault.enc to start fresh."
            ) from exc
        return json.loads(decrypted.decode("utf-8"))

    def set_secret(self, credential_ref: str, secret_value: str) -> None:
        """Encrypt and store a secret by symbolic reference name.

        Refuses to overwrite a vault that cannot be decrypted (prevents silent loss of all stored secrets).
        """
        store = self._read_all()
        store[credential_ref] = secret_value
        payload = json.dumps(store).encode("utf-8")
        tmp = self.vault_file.with_suffix(".enc.tmp")
        tmp.write_bytes(self._fernet.encrypt(payload))
        os.replace(tmp, self.vault_file)

    def get_secret(self, credential_ref: str) -> str | None:
        """Retrieve a secret from OS env vars first, then fallback to encrypted vault."""
        env_val = os.environ.get(credential_ref)
        if env_val:
            return env_val
        try:
            return self._read_all().get(credential_ref)
        except VaultDecryptionError as exc:
            _LOGGER.error("%s", exc)
            return None


class VaultDecryptionError(RuntimeError):
    """Raised when the encrypted credential vault cannot be decrypted with the active master key."""
