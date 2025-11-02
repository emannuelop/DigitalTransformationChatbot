from __future__ import annotations
import sys
from pathlib import Path
import streamlit as st

# ===== Caminho raiz p/ imports do projeto (mantido do original) =====
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ===== Imports locais do pacote =====
from chatbot.ui import db
from chatbot.ui.styles import inject_base_css
from chatbot.ui.sidebar import render_sidebar
from chatbot.ui.views_chat import chat_screen
from chatbot.ui.views_profile import profile_screen

# ---------- Config ----------
st.set_page_config(page_title="Chatbot", page_icon="🤖", layout="wide")
db.init_db()

# ---------- Estado ----------
ss = st.session_state
ss.setdefault("auth_user", None)
ss.setdefault("page", "chat")             # "chat" | "profile"
ss.setdefault("selected_chat_id", None)
ss.setdefault("messages", [])             # cache do chat atual

# ---------- Router ----------
inject_base_css()
if ss["auth_user"] is None:
    from chatbot.ui.views_chat import login_screen  # evita import circular
    login_screen()
else:
    render_sidebar(ss["auth_user"])
    if ss["page"] == "profile":
        profile_screen(ss["auth_user"])
    elif ss["page"] == "ingest":
        from chatbot.ui.views_ingest import ingest_screen
        ingest_screen(ss["auth_user"])
    else:
        chat_screen(ss["auth_user"])

