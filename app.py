"""Data_Talk: Application Entrypoint.

Supports two launch modes:
1. Standalone Production Web Application (Recommended):
       python server.py
   Opens at: http://localhost:8000

2. Via Streamlit (`python -m streamlit run app.py`):
   Automatically starts the FastAPI production backend on port 8000 in a background daemon
   and renders the full-bleed Production Enterprise Web UI inside the viewport.
"""

from __future__ import annotations

import socket
import threading
import time

import streamlit as st
import streamlit.components.v1 as components
import uvicorn

st.set_page_config(
    page_title="Data_Talk — Enterprise Conversational BI & Semantic Studio",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def _is_port_open(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.4)
        return sock.connect_ex((host, port)) == 0


@st.cache_resource
def ensure_production_api_server(port: int = 8504) -> str:
    """Start the FastAPI production server on a background daemon thread if not already running."""
    host = "127.0.0.1"
    if not _is_port_open(host, port):
        from server import app as fastapi_app

        config = uvicorn.Config(fastapi_app, host=host, port=port, log_level="warning")
        server = uvicorn.Server(config)
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()
        for _ in range(25):
            if _is_port_open(host, port):
                break
            time.sleep(0.15)
    return f"http://{host}:{port}"


def main() -> None:
    api_url = ensure_production_api_server(8504)

    # Remove default Streamlit padding, header, and chrome so the Production Web UI fills 100% of the viewport
    st.markdown(
        """
        <style>
        #MainMenu, header, footer, [data-testid="stToolbar"], [data-testid="stHeader"] {
            display: none !important;
        }
        .block-container {
            padding: 0 !important;
            max-width: 100% !important;
        }
        .stApp {
            background-color: #090d16 !important;
            overflow: hidden;
        }
        iframe {
            border: none !important;
            width: 100vw !important;
            height: 100vh !important;
            display: block !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    components.iframe(api_url, height=960, scrolling=True)


if __name__ == "__main__":
    main()
