from __future__ import annotations
import streamlit as st
from .helpers import send_and_respond, ensure_chat_selected
from .styles import inject_base_css
from . import db

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
                    st.success("Conta criada! Faça login na aba ‘Entrar’.")
                except Exception as e:
                    st.error(f"Falha ao cadastrar: {e}")

def chat_screen(user: dict):
    inject_base_css()
    ss = st.session_state

    if ss.get("selected_chat_id") is None:
        ensure_chat_selected(user)

    # Empty state
    if not ss["messages"]:
        st.markdown(
            """
            <div class="hero-wrap">
              <div class="hero">
                <h1>O que quer fazer hoje?</h1>
                <p>Pergunte qualquer coisa</p>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        user_prompt = st.chat_input("Pergunte algo sobre a base de conhecimento…")
        if user_prompt:
            question = user_prompt.strip()
            st.chat_message("user").write(question)
            answer_text, urls, debug = send_and_respond(user, question)
            m = st.chat_message("assistant")
            m.write(answer_text)
            if urls:
                with st.expander("Fontes (links)"):
                    for u in urls:
                        st.markdown(f"- {u}")
            if debug:
                with st.expander("Detalhes técnicos"):
                    st.code(debug)
        return

    # Conversa existente
    for msg in ss["messages"]:
        if msg["role"] == "user":
            st.chat_message("user").write(msg["text"])
        else:
            m = st.chat_message("assistant")
            m.write(msg["text"])
            if msg.get("urls"):
                with st.expander("Fontes (links)"):
                    for u in msg["urls"]:
                        st.markdown(f"- {u}")

    # Input
    user_prompt = st.chat_input("Pergunte algo sobre a base de conhecimento…")
    if user_prompt and ss.get("selected_chat_id") is not None:
        question = user_prompt.strip()
        st.chat_message("user").write(question)
        answer_text, urls, debug = send_and_respond(user, question)
        m = st.chat_message("assistant")
        m.write(answer_text)
        if urls:
            with st.expander("Fontes (links)"):
                for u in urls:
                    st.markdown(f"- {u}")
        if debug:
            with st.expander("Detalhes técnicos"):
                st.code(debug)
