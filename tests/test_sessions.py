"""Tests for multi-session state isolation (workspaces, chat history, overrides)."""

from fastapi.testclient import TestClient

from src.security.auth import SESSION_COOKIE


def test_session_state_isolation(test_client: TestClient):
    # 1. Establish Session A on its own browser client
    client_a = test_client
    resp_a = client_a.get("/")
    assert resp_a.status_code == 200
    cookie_a = resp_a.cookies[SESSION_COOKIE]
    client_a.cookies.set(SESSION_COOKIE, cookie_a)

    # Switch Session A to workspace 'workspace_alpha'
    res_a_set = client_a.post(
        "/api/settings",
        json={"workspace_id": "workspace_alpha", "llm_model": "gpt-4o"},
    )
    assert res_a_set.status_code == 200
    assert res_a_set.json()["active_workspace_id"] == "workspace_alpha"
    assert res_a_set.json()["llm_model"] == "gpt-4o"

    # 2. Establish Session B on a second independent browser client
    client_b = TestClient(test_client.app)
    auth = test_client.app.state.authenticator
    cookie_b = auth.new_session_cookie()
    client_b.cookies.set(SESSION_COOKIE, cookie_b)

    # Switch Session B to workspace 'workspace_beta'
    res_b_set = client_b.post(
        "/api/settings",
        json={"workspace_id": "workspace_beta", "llm_model": "groq/openai/gpt-oss-120b"},
    )
    assert res_b_set.status_code == 200
    assert res_b_set.json()["active_workspace_id"] == "workspace_beta"
    assert res_b_set.json()["llm_model"] == "groq/openai/gpt-oss-120b"

    # 3. Verify Session A is still on 'workspace_alpha' with 'gpt-4o'
    check_a = client_a.get("/api/state")
    assert check_a.status_code == 200
    assert check_a.json()["active_workspace_id"] == "workspace_alpha"
    assert check_a.json()["llm_model"] == "gpt-4o"

    # 4. Clear chat in Session A only affects Session A
    clr_a = client_a.post("/api/chat/clear")
    assert clr_a.status_code == 200
    assert clr_a.json()["active_workspace_id"] == "workspace_alpha"
