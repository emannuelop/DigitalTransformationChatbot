from __future__ import annotations
import streamlit as st
from .helpers import send_and_respond, ensure_chat_selected
from .styles import inject_base_css
from . import db

# ---------------------------------------------
# SUGESTÕES PADRÃO (mostradas apenas em chat novo)
# ---------------------------------------------
SUGGESTED_QUESTIONS = [
    "O que é transformação digital?",
    "Quais são os benefícios da transformação digital?",
    "Como começo um projeto de transformação digital?",
    "Quais indicadores posso usar para medir produtividade?",
]

# Texto padrão quando não há base
_NOT_FOUND = "Não encontrei essa informação na base de conhecimento."

def login_screen():
    inject_base_css()
    tabs = st.tabs(["🔐 Entrar", "🆕 Cadastrar"])

    # Entrar
    with tabs[0]:
        with st.form("login_form", clear_on_submit=False):
            st.subheader("Acesse sua conta")
            email = st.text_input("E-mail", placeholder="voce@exemplo.com")
            password = st.text_input("Senha", type="password")
            submitted = st.form_submit_button("Entrar", use_container_width=True)
        if submitted:
            user = db.authenticate(email, password)
            if user:
                st.session_state["auth_user"] = user
                st.session_state["selected_chat_id"] = None
                st.session_state["messages"].clear()
                ensure_chat_selected(user)
                st.success(f"Bem-vindo(a), {user['name']}!")
                st.rerun()
            else:
                st.error("Credenciais inválidas.")

    # Cadastrar
    with tabs[1]:
        with st.form("signup_form", clear_on_submit=False):
            st.subheader("Crie sua conta")
            name = st.text_input("Nome completo")
            email2 = st.text_input("E-mail")
            pwd1 = st.text_input("Senha", type="password")
            pwd2 = st.text_input("Confirmar senha", type="password")
            submitted2 = st.form_submit_button("Cadastrar", use_container_width=True)
        if submitted2:
            if not name.strip() or not email2.strip() or not pwd1:
                st.error("Preencha todos os campos.")
            elif pwd1 != pwd2:
                st.error("As senhas não conferem.")
            else:
                try:
                    _ = db.create_user(name=name.strip(), email=email2.strip(), password=pwd1)
                    st.success("Conta criado! Faça login na aba ‘Entrar’.")
                except Exception as e:
                    st.error(f"Falha ao cadastrar: {e}")

def chat_screen(user: dict):
    inject_base_css()
    ss = st.session_state

    if ss.get("selected_chat_id") is None:
        ensure_chat_selected(user)

    # Estado vazio (chat novo)
    if not ss["messages"]:
        st.markdown(
            """
            <div class="hero-wrap">
              <div class="hero">
                <h1>O que quer fazer hoje?</h1>
                <p>Pergunte sobre transformação digital coisa ou escolha uma sugestão abaixo.</p>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        sugg_box = st.container()
        with sugg_box:
            st.markdown('<div class="sugg">', unsafe_allow_html=True)
            st.markdown("#### Sugestões")
            cols = st.columns(2, vertical_alignment="center")
            clicked = None
            for i, q in enumerate(SUGGESTED_QUESTIONS):
                if cols[i % 2].button(q, key=f"sugg_{i}", use_container_width=True):
                    clicked = q
            st.markdown("</div>", unsafe_allow_html=True)

        if clicked:
            question = clicked.strip()
            sugg_box.empty()
            st.chat_message("user").write(question)
            send_and_respond(user, question)
            st.rerun()

        user_prompt = st.chat_input("Pergunte algo sobre a base de conhecimento…")
        if user_prompt:
            question = user_prompt.strip()
            st.chat_message("user").write(question)
            send_and_respond(user, question)
            st.rerun()
        return

    # Conversa existente
    for msg in ss["messages"]:
        if msg["role"] == "user":
            st.chat_message("user").write(msg["text"])
        else:
            m = st.chat_message("assistant")
            m.write(msg["text"])

            # ---- Fontes (links) — mostrar somente se houver urls e NÃO for mensagem de 'não encontrado'
            if msg.get("urls") and (msg.get("text", "").strip() != _NOT_FOUND):
                from urllib.parse import urlparse

                def _pretty_label(u: str, max_len: int = 90) -> str:
                    if not isinstance(u, str) or not u:
                        return u
                    if not u.startswith(("http://", "https://")):
                        return u
                    p = urlparse(u)
                    label = (p.netloc or "") + (p.path or "")
                    if p.query:
                        label += "?" + p.query
                    if len(label) > max_len:
                        label = label[: max_len - 1] + "…"
                    return label

                with st.expander("Fontes (links)"):
                    for u in msg["urls"]:
                        if isinstance(u, str) and u.startswith(("http://", "https://")):
                            label = _pretty_label(u)
                            st.markdown(f"- [{label}]({u})")
                        else:
                            st.markdown(f"- {u}")

    user_prompt = st.chat_input("Pergunte algo sobre a base de conhecimento…")
    if user_prompt and ss.get("selected_chat_id") is not None:
        question = user_prompt.strip()
        st.chat_message("user").write(question)
        send_and_respond(user, question)
        st.rerun()
