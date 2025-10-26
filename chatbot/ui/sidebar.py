from __future__ import annotations
import streamlit as st
from .styles import inject_base_css
from . import db
from .helpers import (
    ensure_chat_selected, load_messages_into_state,
    get_user_photo_path, img_tag
)

def render_sidebar(user: dict) -> None:
    ensure_chat_selected(user)
    with st.sidebar:
        inject_base_css()  # tema + ajustes

        # Novo chat
        if st.button("Novo chat", use_container_width=True):
            cid = db.create_chat(user["id"], "Novo chat")
            st.session_state["selected_chat_id"] = cid
            st.session_state["messages"].clear()
            st.session_state["page"] = "chat"
            st.rerun()

        # >>> NOVO BOTÃO (apenas navegação de página) <<<
        if st.button("➕ Adicionar base de conhecimento", use_container_width=True):
            st.session_state["page"] = "ingest"
            st.rerun()

        # Histórico
        st.markdown('<div class="sb-section">Histórico</div>', unsafe_allow_html=True)
        chats = db.list_chats(user["id"])
        if not chats:
            st.caption("Sem conversas ainda.")
        else:
            for chat in chats:
                cols = st.columns([0.82, 0.18], vertical_alignment="center")
                with cols[0]:
                    active = (chat["id"] == st.session_state["selected_chat_id"])
                    label = f"● {chat['title']}" if active else chat["title"]
                    if st.button(label, key=f"sel_{chat['id']}", use_container_width=True, help=chat["title"]):
                        st.session_state["selected_chat_id"] = chat["id"]
                        load_messages_into_state(user["id"], chat["id"])
                        st.session_state["page"] = "chat"
                        st.rerun()
                with cols[1]:
                    with st.popover("", use_container_width=True):
                        new_title = st.text_input("Renomear", value=chat["title"], key=f"rn_{chat['id']}")
                        if st.button("Salvar nome", key=f"rns_{chat['id']}", use_container_width=True):
                            db.rename_chat(user["id"], chat["id"], new_title)
                            st.rerun()
                        if st.button("Excluir chat", key=f"del_{chat['id']}", use_container_width=True):
                            db.delete_chat(user["id"], chat["id"])
                            st.session_state["selected_chat_id"] = None
                            st.session_state["messages"].clear()
                            ensure_chat_selected(user)
                            st.session_state["page"] = "chat"
                            st.rerun()

        # Rodapé (userbar)
        st.markdown('<div class="sb-userbar">', unsafe_allow_html=True)
        ucols = st.columns([0.18, 0.82], vertical_alignment="center")

        photo_path = get_user_photo_path(user["id"])
        with ucols[0]:
            if photo_path and photo_path.exists():
                st.markdown(img_tag(photo_path, 30, "sb-avatar-img"), unsafe_allow_html=True)
            else:
                avatar_letter = (user["name"][:1] or "U").upper()
                st.markdown(f'<div class="sb-avatar">{avatar_letter}</div>', unsafe_allow_html=True)

        with ucols[1]:
            with st.popover(user["name"], use_container_width=True):
                if st.button("Configurações", use_container_width=True):
                    st.session_state["page"] = "profile"
                    st.rerun()
                if st.button("Sair", use_container_width=True):
                    st.session_state["auth_user"] = None
                    st.session_state["messages"].clear()
                    st.session_state["selected_chat_id"] = None
                    st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
