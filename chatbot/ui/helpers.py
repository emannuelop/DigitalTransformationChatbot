# chatbot/ui/helpers.py
from __future__ import annotations
from pathlib import Path
from glob import glob
import base64
import mimetypes
from typing import List, Tuple, Dict
import sqlite3
import streamlit as st
import json

from chatbot.ml import rag_pipeline as rp
from . import db

NOT_FOUND_TEXT = getattr(
    rp, "NOT_FOUND_TEXT",
    "Não encontrei essa informação na base de conhecimento."
)

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

# ---------- Conexão com knowledge_base (para citar fontes pelo nome) ----------
def _kb_conn() -> sqlite3.Connection:
    # Import tardio para evitar conflitos de path
    from chatbot.extraction.scraping import DB_PATH
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

# --- nova versão (com filtro por usuário) ---
def _lookup_user_labels(urls: List[str], user_id: int | None) -> Dict[str, str]:
    """
    Retorna url -> display_name SOMENTE das fontes que este usuário anexou (user_sources.user_id = user_id).
    Evita mostrar o nome do PDF de outra conta mesmo que a URL coincida.
    """
    if not urls or user_id is None:
        return {}
    try:
        conn = _kb_conn()
        placeholders = ",".join(["?"] * len(urls))
        rows = conn.execute(f"""
            SELECT d.url, us.display_name
            FROM user_sources us
            JOIN documents d ON d.id = us.doc_id
            WHERE d.url IN ({placeholders})
              AND us.user_id = ?
        """, urls + [user_id]).fetchall()
        conn.close()
        return {r["url"]: r["display_name"] for r in rows}
    except Exception:
        return {}

def unique_urls_in_order(df, limit: int = 5, user_id: int | None = None) -> List[str]:
    """
    Igual à anterior, mas garante que só rotulamos com display_name de PDFs do PRÓPRIO usuário.
    """
    seen, raw_urls = set(), []
    if df is None:
        return []
    for _, r in df.iterrows():
        u = (r.get("url") or "").strip()
        if u and u not in seen:
            seen.add(u)
            raw_urls.append(u)
        if len(raw_urls) >= limit:
            break

    labels = _lookup_user_labels(raw_urls, user_id=user_id)
    pretty = []
    for u in raw_urls:
        if u in labels:
            pretty.append(f"{labels[u]} — fonte (PDF)")
        else:
            pretty.append(u)
    return pretty

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

def call_rag(question: str, user_id: int) -> Tuple[str, List[str], str | None]:
    # Não monte fontes ainda; só recupere o contexto
    urls: List[str] = []
    debug: str | None = None
    ctx = None
    try:
        index, mapping = cached_search_handles()
        ctx = search(index, mapping, question, user_id=user_id, k=cfg.TOP_K)
    except Exception as e:
        debug = f"Falha ao recuperar contexto: {e}"
        ctx = None

    # Geração com os gates do pipeline
    try:
        out = rag_pipeline.answer_with_cfg(
            question, user_id=user_id, gen_overrides=None, k=cfg.TOP_K
        )
        if isinstance(out, tuple):
            answer_text = out[0] if len(out) > 0 else ""
            # out[1] são URLs brutas do pipeline (opcional); out[2] é o motivo/debug
            if len(out) > 2 and out[2]:
                debug = f"{debug + chr(10) if debug else ''}{out[2]}".strip()
        else:
            answer_text = str(out or "")
    except Exception as e:
        # Erro na geração: devolva sem fontes
        answer_text = "Tive um problema para gerar a resposta agora. Tente novamente em instantes."
        return answer_text, [], f"{debug + chr(10) if debug else ''}{e}".strip()

    # Só mostre fontes se a resposta for válida (nem NOT_FOUND, nem msg de erro)
    bad = (
        not answer_text
        or answer_text.strip() == NOT_FOUND_TEXT
        or answer_text.startswith("Tive um problema")
    )
    if bad:
        return answer_text, [], debug

    # Resposta válida → derive as fontes do contexto final
    if ctx is not None and hasattr(ctx, "empty") and not ctx.empty:
        urls = unique_urls_in_order(ctx, limit=5, user_id=user_id)
    else:
        urls = []
    return answer_text, urls, debug

def send_and_respond(user: dict, question: str):
    ss = st.session_state
    db.save_message(user["id"], "user", question, chat_id=ss["selected_chat_id"])
    ss["messages"].append({"role": "user", "text": question, "urls": []})
    user_id = user["id"]
    with st.spinner("Consultando base e gerando resposta..."):
        answer_text, urls, debug = call_rag(question, user_id)
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
