from __future__ import annotations
import sys
from pathlib import Path
from glob import glob
import base64
import mimetypes
import streamlit as st

# Caminho raiz p/ imports do projeto
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Pipeline RAG (fallback seguro se não estiver instalado)
try:
    from chatbot.ml import settings as cfg
    from chatbot.ml import rag_pipeline
    from chatbot.ml.rag_pipeline import load_search, search
except Exception:
    cfg = type("cfg", (), {"TOP_K": 5})
    def load_search(): return (None, None)
    def search(*args, **kwargs): return None
    class _Dummy:
        @staticmethod
        def answer_with_cfg(q, gen_overrides=None, k=5):
            return ("Não consegui consultar o modelo agora. Tente novamente em instantes.", [])
    rag_pipeline = _Dummy()

# DB e estilos
from chatbot.ui import db
from chatbot.ui.styles import inject_base_css

# ---------- Config ----------
st.set_page_config(page_title="Chatbot • RAG", page_icon="🤖", layout="wide")
db.init_db()

# ---------- Cache RAG ----------
@st.cache_resource(show_spinner=False)
def _cached_search_handles():
    return load_search()

def _unique_urls_in_order(df, limit=5):
    seen, out = set(), []
    if df is None:
        return out
    for _, r in df.iterrows():
        u = (r.get("url") or "").strip()
        if u and u not in seen:
            seen.add(u); out.append(u)
        if len(out) >= limit: break
    return out

def _call_rag(question: str):
    urls, debug = [], None
    try:
        index, mapping = _cached_search_handles()
        ctx = search(index, mapping, question, k=cfg.TOP_K)
        urls = _unique_urls_in_order(ctx, limit=5)
    except Exception as e:
        debug = f"Falha ao recuperar fontes: {e}"
    try:
        out = rag_pipeline.answer_with_cfg(question, gen_overrides=None, k=cfg.TOP_K)
        answer_text = out[0] if isinstance(out, tuple) else out
        if not isinstance(answer_text, str):
            answer_text = str(answer_text)
    except Exception as e:
        debug = f"{debug or ''}\n{e}".strip()
        answer_text = "Tive um problema para gerar a resposta agora. Tente novamente em instantes."
    return answer_text, urls, debug

# ---------- Estado ----------
ss = st.session_state
ss.setdefault("auth_user", None)
ss.setdefault("page", "chat")             # "chat" | "profile"
ss.setdefault("selected_chat_id", None)
ss.setdefault("messages", [])             # cache do chat atual

# ---------- Foto do usuário ----------
def _photo_dir():
    d = db.DATA_DIR / "user_photos"
    d.mkdir(parents=True, exist_ok=True)
    return d

def _get_user_photo_path(user_id: int) -> Path | None:
    files = []
    for p in [str(_photo_dir() / f"{user_id}.*")]:
        files.extend(glob(p))
    if not files:
        return None
    files.sort(key=lambda p: Path(p).stat().st_mtime, reverse=True)
    return Path(files[0])

def _save_user_photo(user_id: int, uploaded_file) -> Path:
    ext = Path(uploaded_file.name).suffix.lower() or ".png"
    if ext not in (".png", ".jpg", ".jpeg", ".webp"):
        ext = ".png"
    for old in _photo_dir().glob(f"{user_id}.*"):
        try: old.unlink()
        except Exception: pass
    out = _photo_dir() / f"{user_id}{ext}"
    out.write_bytes(uploaded_file.getbuffer())
    return out

def _img_tag(path: Path, size: int, css_class: str = "") -> str:
    """Retorna um <img> base64 com tamanho fixo (quadrado) — ideal p/ avatar redondo via CSS."""
    if not path or not path.exists():
        return ""
    mime = mimetypes.guess_type(str(path))[0] or "image/png"
    b64 = base64.b64encode(path.read_bytes()).decode()
    cls = f' class="{css_class}"' if css_class else ""
    return f'<img src="data:{mime};base64,{b64}" width="{size}" height="{size}"{cls} />'

# ---------- Auxiliares ----------
def _load_messages_into_state(user_id: int, chat_id: int) -> None:
    ss["messages"] = []
    for role, text, _ts in db.load_history(user_id, limit=500, chat_id=chat_id):
        ss["messages"].append({"role": role, "text": text, "urls": []})

def _ensure_chat_selected(user: dict) -> None:
    chats = db.list_chats(user["id"])
    if not chats:
        cid = db.create_chat(user["id"], "Novo chat")
        ss["selected_chat_id"] = cid
        _load_messages_into_state(user["id"], cid)
        return
    if ss.get("selected_chat_id") is None:
        ss["selected_chat_id"] = chats[0]["id"]
        _load_messages_into_state(user["id"], ss["selected_chat_id"])

def _title_from_prompt(text: str, max_len: int = 40) -> str:
    t = text.strip().replace("\n", " ")
    return (t[:max_len] + "…") if len(t) > max_len else (t or "Sem título")

def _send_and_respond(user: dict, question: str):
    db.save_message(user["id"], "user", question, chat_id=ss["selected_chat_id"])
    ss["messages"].append({"role": "user", "text": question, "urls": []})
    with st.spinner("Consultando base e gerando resposta..."):
        answer_text, urls, debug = _call_rag(question)
    ss["messages"].append({"role": "assistant", "text": answer_text, "urls": urls})
    db.save_message(user["id"], "assistant", answer_text, chat_id=ss["selected_chat_id"])
    chats = db.list_chats(user["id"])
    cur = next((c for c in chats if c["id"] == ss["selected_chat_id"]), None)
    if cur and (cur["title"] == "Novo chat" or cur["title"] == "Sem título"):
        try:
            db.rename_chat(user["id"], ss["selected_chat_id"], _title_from_prompt(question))
        except Exception:
            pass
    return answer_text, urls, debug

# ---------- Login / Cadastro ----------
def _login_screen():
    tabs = st.tabs(["🔐 Entrar", "🆕 Cadastrar"])

    with tabs[0]:
        with st.form("login_form", clear_on_submit=False):
            st.subheader("Acesse sua conta")
            email = st.text_input("E-mail", placeholder="voce@exemplo.com")
            password = st.text_input("Senha", type="password")
            submitted = st.form_submit_button("Entrar", use_container_width=True)
        if submitted:
            user = db.authenticate(email, password)
            if user:
                ss["auth_user"] = user
                ss["selected_chat_id"] = None
                ss["messages"].clear()
                _ensure_chat_selected(user)
                st.success(f"Bem-vindo(a), {user['name']}!")
                st.rerun()
            else:
                st.error("Credenciais inválidas.")

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

# ---------- Sidebar ----------
def _sidebar(user: dict):
    _ensure_chat_selected(user)
    with st.sidebar:
        inject_base_css()  # tema + ajustes

        # Novo chat (grudado no topo, perto da seta da sidebar)
        if st.button("Novo chat", use_container_width=True):
            cid = db.create_chat(user["id"], "Novo chat")
            ss["selected_chat_id"] = cid
            ss["messages"].clear()
            ss["page"] = "chat"
            st.rerun()

        # Histórico
        st.markdown('<div class="sb-section">Histórico</div>', unsafe_allow_html=True)
        chats = db.list_chats(user["id"])
        if not chats:
            st.caption("Sem conversas ainda.")
        else:
            for chat in chats:
                # Coluna da seta com largura suficiente para não "sumir"
                cols = st.columns([0.82, 0.18], vertical_alignment="center")
                with cols[0]:
                    active = (chat["id"] == ss["selected_chat_id"])
                    label = f"● {chat['title']}" if active else chat["title"]
                    if st.button(label, key=f"sel_{chat['id']}", use_container_width=True, help=chat["title"]):
                        ss["selected_chat_id"] = chat["id"]
                        _load_messages_into_state(user["id"], chat["id"])
                        ss["page"] = "chat"
                        st.rerun()
                with cols[1]:
                    # O botão de popover fica sempre visível via CSS (min-width/height fixos)
                    with st.popover("", use_container_width=True):
                        new_title = st.text_input("Renomear", value=chat["title"], key=f"rn_{chat['id']}")
                        if st.button("Salvar nome", key=f"rns_{chat['id']}", use_container_width=True):
                            db.rename_chat(user["id"], chat["id"], new_title)
                            st.rerun()
                        if st.button("Excluir chat", key=f"del_{chat['id']}", use_container_width=True):
                            db.delete_chat(user["id"], chat["id"])
                            ss["selected_chat_id"] = None
                            ss["messages"].clear()
                            _ensure_chat_selected(user)
                            ss["page"] = "chat"
                            st.rerun()

        # ===== Userbar fixa no rodapé =====
        st.markdown('<div class="sb-userbar">', unsafe_allow_html=True)
        ucols = st.columns([0.18, 0.82], vertical_alignment="center")

        # Avatar (foto se houver; senão inicial)
        photo_path = _get_user_photo_path(user["id"])
        with ucols[0]:
            if photo_path and photo_path.exists():
                st.markdown(_img_tag(photo_path, 30, "sb-avatar-img"), unsafe_allow_html=True)
            else:
                avatar_letter = (user["name"][:1] or "U").upper()
                st.markdown(f'<div class="sb-avatar">{avatar_letter}</div>', unsafe_allow_html=True)

        with ucols[1]:
            # Nome visível; menu ao clicar
            with st.popover(user["name"], use_container_width=True):
                if st.button("Configurações", use_container_width=True):
                    ss["page"] = "profile"
                    st.rerun()
                if st.button("Sair", use_container_width=True):
                    ss["auth_user"] = None
                    ss["messages"].clear()
                    ss["selected_chat_id"] = None
                    st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

# ---------- Páginas ----------
def _chat_screen(user: dict):
    inject_base_css()

    if ss.get("selected_chat_id") is None:
        _ensure_chat_selected(user)

    # ===== Empty state (padronizado com st.chat_input) =====
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
            answer_text, urls, debug = _send_and_respond(user, question)
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

    # ===== Conversa já iniciada =====
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

    # Entrada estilo ChatGPT para continuar
    user_prompt = st.chat_input("Pergunte algo sobre a base de conhecimento…")
    if user_prompt and ss.get("selected_chat_id") is not None:
        question = user_prompt.strip()
        st.chat_message("user").write(question)
        answer_text, urls, debug = _send_and_respond(user, question)
        m = st.chat_message("assistant")
        m.write(answer_text)
        if urls:
            with st.expander("Fontes (links)"):
                for u in urls:
                    st.markdown(f"- {u}")
        if debug:
            with st.expander("Detalhes técnicos"):
                st.code(debug)

def _profile_screen(user: dict):
    inject_base_css()

    st.subheader("Perfil do usuário")

    # Mostra foto atual (se houver) + e-mail (somente leitura)
    current = _get_user_photo_path(user["id"])
    cols_top = st.columns([0.22, 0.78])
    with cols_top[0]:
        if current and current.exists():
            st.markdown(_img_tag(current, 96, "sb-avatar-img-96"), unsafe_allow_html=True)
        else:
            st.markdown('<div class="sb-avatar" style="width:96px;height:96px;font-size:1.8rem;">'
                        f'{(user["name"][:1] or "U").upper()}</div>', unsafe_allow_html=True)
    with cols_top[1]:
        st.text_input("E-mail", value=user["email"], disabled=True)

    st.markdown("---")

    # Alterar nome
    with st.form("profile_name"):
        name = st.text_input("Nome", value=user["name"])
        ok = st.form_submit_button("Salvar nome")
    if ok:
        try:
            db.update_user_name(user["id"], name.strip())
            ss["auth_user"]["name"] = name.strip()
            st.success("Nome atualizado com sucesso.")
        except Exception as e:
            st.error(f"Falha ao atualizar nome: {e}")

    # Upload/alterar foto
    st.markdown("### Foto do usuário")
    up = st.file_uploader("Envie uma imagem (png/jpg/jpeg/webp)", type=["png", "jpg", "jpeg", "webp"])
    colu = st.columns([0.25, 0.75])
    with colu[0]:
        if up is not None and up.size > 0:
            try:
                saved = _save_user_photo(user["id"], up)
                st.success("Foto atualizada.")
                st.markdown(_img_tag(saved, 96, "sb-avatar-img-96"), unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Falha ao salvar foto: {e}")

    st.markdown("---")

    # Trocar senha
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

# ---------- Router ----------
if ss["auth_user"] is None:
    _login_screen()
else:
    _sidebar(ss["auth_user"])
    if ss["page"] == "profile":
        _profile_screen(ss["auth_user"])
    else:
        _chat_screen(ss["auth_user"])
