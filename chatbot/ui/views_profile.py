from __future__ import annotations
import streamlit as st
from .styles import inject_base_css
from .helpers import get_user_photo_path, save_user_photo, img_tag
from . import db

def profile_screen(user: dict):
    inject_base_css()

    st.subheader("Perfil do usuário")

    # Foto + e-mail
    current = get_user_photo_path(user["id"])
    cols_top = st.columns([0.22, 0.78])
    with cols_top[0]:
        if current and current.exists():
            st.markdown(img_tag(current, 96, "sb-avatar-img-96"), unsafe_allow_html=True)
        else:
            st.markdown('<div class="sb-avatar" style="width:96px;height:96px;font-size:1.8rem;">'
                        f'{(user["name"][:1] or "U").upper()}</div>', unsafe_allow_html=True)
    with cols_top[1]:
        st.text_input("E-mail", value=user["email"], disabled=True)

    st.markdown("---")

    # Nome
    with st.form("profile_name"):
        name = st.text_input("Nome", value=user["name"])
        ok = st.form_submit_button("Salvar nome")
    if ok:
        try:
            db.update_user_name(user["id"], name.strip())
            st.session_state["auth_user"]["name"] = name.strip()
            st.success("Nome atualizado com sucesso.")
        except Exception as e:
            st.error(f"Falha ao atualizar nome: {e}")

    # Foto
    st.markdown("### Foto do usuário")
    up = st.file_uploader("Envie uma imagem (png/jpg/jpeg/webp)", type=["png", "jpg", "jpeg", "webp"])
    colu = st.columns([0.25, 0.75])
    with colu[0]:
        if up is not None and up.size > 0:
            try:
                saved = save_user_photo(user["id"], up)
                st.success("Foto atualizada.")
                st.markdown(img_tag(saved, 96, "sb-avatar-img-96"), unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Falha ao salvar foto: {e}")

    st.markdown("---")

    # Senha
    st.subheader("Trocar senha")
    with st.form("profile_pwd"):
        old = st.text_input("Senha atual", type="password")
        new1 = st.text_input("Nova senha", type="password")
        new2 = st.text_input("Confirmar nova senha", type="password")
        ok2 = st.form_submit_button("Atualizar senha")
    if ok2:
        if not old or not new1 or not new2:
            st.error("Preencha todos os campos.")
        elif new1 != new2:
            st.error("As novas senhas não conferem.")
        else:
            try:
                db.update_user_password(user["id"], old, new1)
                st.success("Senha atualizada com sucesso.")
            except ValueError as ve:
                st.error(str(ve))
            except Exception as e:
                st.error(f"Falha ao atualizar senha: {e}")
