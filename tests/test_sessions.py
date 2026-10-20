"""Tests for multi-session state isolation (workspaces, chat history, overrides)."""

from fastapi.testclient import TestClient

from src.security.auth import SESSION_COOKIE


def test_session_state_isolation(test_client: TestClient):
    # 1. Establish Session A
    resp_a = test_client.get("/")
    assert resp_a.status_code == 200
    cookie_a = resp_a.cookies[SESSION_COOKIE]

    # Switch Session A to workspace 'workspace_alpha'
    res_a_set = test_client.post(
        "/api/settings",
        json={"workspace_id": "workspace_alpha", "llm_model": "gpt-4o"},
        cookies={SESSION_COOKIE: cookie_a},
    )
    assert res_a_set.status_code == 200
    assert res_a_set.json()["active_workspace_id"] == "workspace_alpha"
    assert res_a_set.json()["llm_model"] == "gpt-4o"

    # 2. Establish Session B with a fresh session cookie
    auth = test_client.app.state.authenticator
    cookie_b = auth.new_session_cookie()

    # Switch Session B to workspace 'workspace_beta'
    res_b_set = test_client.post(
        "/api/settings",
        json={"workspace_id": "workspace_beta", "llm_model": "groq/openai/gpt-oss-120b"},
        cookies={SESSION_COOKIE: cookie_b},
    )
    assert res_b_set.status_code == 200
    assert res_b_set.json()["active_workspace_id"] == "workspace_beta"
    assert res_b_set.json()["llm_model"] == "groq/openai/gpt-oss-120b"

    # 3. Verify Session A is still on 'workspace_alpha' with 'gpt-4o'
    check_a = test_client.get("/api/state", cookies={SESSION_COOKIE: cookie_a})
    assert check_a.status_code == 200
    assert check_a.json()["active_workspace_id"] == "workspace_alpha"
    assert check_a.json()["llm_model"] == "gpt-4o"

    # 4. Clear chat in Session A only affects Session A
    clr_a = test_client.post("/api/chat/clear", cookies={SESSION_COOKIE: cookie_a})
    assert clr_a.status_code == 200
    assert clr_a.json()["active_workspace_id"] == "workspace_alpha"
