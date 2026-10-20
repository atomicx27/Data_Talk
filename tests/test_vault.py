"""Tests for CredentialVault encryption, key priorities, and error handling."""

from pathlib import Path
from cryptography.fernet import Fernet
import pytest

from src.security.privacy_firewall import CredentialVault, VaultDecryptionError


def test_credential_vault_roundtrip(tmp_path: Path):
    vault = CredentialVault(base_dir=tmp_path / ".secrets")
    vault.set_secret("db_password", "super_secret_value_123")
    assert vault.get_secret("db_password") == "super_secret_value_123"


def test_credential_vault_env_key_priority(tmp_path: Path, monkeypatch):
    custom_key = Fernet.generate_key().decode("utf-8")
    monkeypatch.setenv("DATA_TALK_MASTER_KEY", custom_key)

    vault = CredentialVault(base_dir=tmp_path / ".secrets")
    assert vault.key_source == "env"
    vault.set_secret("api_token", "abc-token")
    assert vault.get_secret("api_token") == "abc-token"


def test_credential_vault_mismatched_key_raises_vault_error(tmp_path: Path, monkeypatch):
    key1 = Fernet.generate_key().decode("utf-8")
    monkeypatch.setenv("DATA_TALK_MASTER_KEY", key1)
    vault1 = CredentialVault(base_dir=tmp_path / ".secrets")
    vault1.set_secret("target_token", "sensitive_data")

    # Change to a different master key
    key2 = Fernet.generate_key().decode("utf-8")
    monkeypatch.setenv("DATA_TALK_MASTER_KEY", key2)

    vault2 = CredentialVault(base_dir=tmp_path / ".secrets")
    # Setting secret should refuse to overwrite an un-decryptable vault
    with pytest.raises(VaultDecryptionError):
        vault2.set_secret("another_token", "fail")
