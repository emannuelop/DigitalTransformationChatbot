from __future__ import annotations
from pathlib import Path
from glob import glob
import base64
import mimetypes
from typing import List, Tuple
import streamlit as st
import json

from . import db

# ---------- Foto do usuário ----------
def photo_dir() -> Path:
    d = db.DATA_DIR / "user_photos"
    d.mkdir(parents=True, exist_ok=True)
    return d

def get_user_photo_path(user_id: int) -> Path | None:
    files = []
    for p in [str(photo_dir() / f"{user_id}.*")]:
        files.extend(glob(p))
    if not files:
        return None
    files.sort(key=lambda p: Path(p).stat().st_mtime, reverse=True)
    return Path(files[0])

def save_user_photo(user_id: int, uploaded_file) -> Path:
    ext = Path(uploaded_file.name).suffix.lower() or ".png"
    if ext not in (".png", ".jpg", ".jpeg", ".webp"):
        ext = ".png"
    for old in photo_dir().glob(f"{user_id}.*"):
        try:
            old.unlink()
        except Exception:
            pass
    out = photo_dir() / f"{user_id}{ext}"
    out.write_bytes(uploaded_file.getbuffer())
    return out

def img_tag(path: Path, size: int, css_class: str = "") -> str:
    if not path or not path.exists():
        return ""
    mime = mimetypes.guess_type(str(path))[0] or "image/png"
    b64 = base64.b64encode(path.read_bytes()).decode()
    cls = f' class="{css_class}"' if css_class else ""
    return f'<img src="data:{mime};base64,{b64}" width="{size}" height="{size}"{cls} />'

# ---------- Títulos/strings ----------
def title_from_prompt(text: str, max_len: int = 40) -> str:
    t = (text or "").strip().replace("\n", " ")
    return (t[:max_len] + "…") if len(t) > max_len else (t or "Sem título")


# ---------- Histórico/estado ----------
def load_messages_into_state(user_id: int, chat_id: int) -> None:
    ss = st.session_state
    ss["messages"] = []
    for role, text, urls, _ts in db.load_history(user_id, limit=500, chat_id=chat_id):
        ss["messages"].append({"role": role, "text": text, "urls": urls})

def ensure_chat_selected(user: dict) -> None:
    ss = st.session_state
    chats = db.list_chats(user["id"])
    if not chats:
        cid = db.create_chat(user["id"], "Novo chat")
        ss["selected_chat_id"] = cid
        load_messages_into_state(user["id"], cid)
        return
    if ss.get("selected_chat_id") is None:
        ss["selected_chat_id"] = chats[0]["id"]
        load_messages_into_state(user["id"], ss["selected_chat_id"])

# ---------- RAG ----------
try:
    from chatbot.ml import settings as cfg
    from chatbot.ml import rag_pipeline
    from chatbot.ml.rag_pipeline import load_search, search
except Exception:
    cfg = type("cfg", (), {"TOP_K": 5})

    def load_search():
        return (None, None)

    def search(*args, **kwargs):
        return None

    class _Dummy:
        @staticmethod
        def answer_with_cfg(q, gen_overrides=None, k=5):
            return ("Não consegui consultar o modelo agora. Tente novamente em instantes.", [])

    rag_pipeline = _Dummy()

@st.cache_resource(show_spinner=False)
def cached_search_handles():
    return load_search()

def unique_urls_in_order(df, limit=5) -> List[str]:
    seen, out = set(), []
    if df is None:
        return out
    for _, r in df.iterrows():
        u = (r.get("url") or "").strip()
        if u and u not in seen:
            seen.add(u)
            out.append(u)
        if len(out) >= limit:
            break
    return out

def call_rag(question: str) -> Tuple[str, List[str], str | None]:
    urls, debug = [], None
    try:
        index, mapping = cached_search_handles()
        ctx = search(index, mapping, question, k=cfg.TOP_K)
        urls = unique_urls_in_order(ctx, limit=5)
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

def send_and_respond(user: dict, question: str):
    ss = st.session_state
    db.save_message(user["id"], "user", question, chat_id=ss["selected_chat_id"])
    ss["messages"].append({"role": "user", "text": question, "urls": []})
    with st.spinner("Consultando base e gerando resposta..."):
        answer_text, urls, debug = call_rag(question)
    ss["messages"].append({"role": "assistant", "text": answer_text, "urls": urls})
    db.save_message(user["id"], "assistant", answer_text, chat_id=ss["selected_chat_id"], urls=urls)

    chats = db.list_chats(user["id"])
    cur = next((c for c in chats if c["id"] == ss["selected_chat_id"]), None)
    if cur and (cur["title"] in ("Novo chat", "Sem título")):
        try:
            db.rename_chat(user["id"], ss["selected_chat_id"], title_from_prompt(question))
        except Exception:
            pass
    return answer_text, urls, debug