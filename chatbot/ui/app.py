# chatbot/ui/app.py
import sys
from pathlib import Path
import streamlit as st

# Garantir que o pacote "chatbot" seja importável quando rodar pelo Streamlit
REPO_ROOT = Path(__file__).resolve().parents[2]  # .../DigitalTransformationChatbot
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Imports do projeto
from chatbot.ml import settings as cfg
from chatbot.ml.rag_pipeline import load_search, search, answer_with_cfg
from chatbot.ml.llm_backends import LMStudioBackend

# -----------------------------
# Cache (carregar índice uma única vez)
# -----------------------------
@st.cache_resource(show_spinner=False)
def cached_load_search():
    return load_search()  # (index, mapping)

# -----------------------------
# Helpers
# -----------------------------
def unique_urls_in_order(df, limit=5):
    seen, out = set(), []
    for _, r in df.iterrows():
        u = (r.get("url") or "").strip()
        if u and u not in seen:
            seen.add(u)
            out.append(u)
        if len(out) >= limit:
            break
    return out

def configure_runtime():
    """Sidebar: lê valores e aplica nos settings do projeto em runtime."""
    st.sidebar.header("⚙️ Configurações")

    # LM Studio
    lm_host = st.sidebar.text_input("LM Studio host", value=str(cfg.LMSTUDIO_HOST))
    lm_model = st.sidebar.text_input("Modelo (ID no LM Studio)", value=str(cfg.LMSTUDIO_MODEL))

    # Checagem de saúde do LM Studio
    with st.sidebar.expander("Saúde do LM Studio", expanded=False):
        client = LMStudioBackend(model=lm_model, host=lm_host)
        health = client.health()
        st.json(health, expanded=False)

    # Recuperação (RAG)
    top_k = st.sidebar.slider("TOP_K (chunks)", min_value=1, max_value=8, value=int(cfg.TOP_K), step=1)

    # Geração (LLM)
    temperature = st.sidebar.slider("Temperature", 0.0, 1.0, float(cfg.GEN_TEMPERATURE), 0.05)
    top_p = st.sidebar.slider("Top-p", 0.1, 1.0, float(cfg.GEN_TOP_P), 0.05)
    max_tokens = st.sidebar.slider("Máx. tokens de saída", 128, 4096, int(cfg.GEN_MAX_TOKENS), 64)
    timeout_s = st.sidebar.slider("Tempo máx. de geração (s)", 5, 120, int(cfg.GEN_TIMEOUT_S), 5)
    retries = st.sidebar.slider("Tentativas (retries)", 1, 3, int(cfg.GEN_RETRIES), 1)

    # Aplicar nos settings globais (usados pelo pipeline)
    cfg.LMSTUDIO_HOST = lm_host
    cfg.LMSTUDIO_MODEL = lm_model
    cfg.TOP_K = int(top_k)
    cfg.GEN_TEMPERATURE = float(temperature)
    cfg.GEN_TOP_P = float(top_p)
    cfg.GEN_MAX_TOKENS = int(max_tokens)
    cfg.GEN_TIMEOUT_S = int(timeout_s)
    cfg.GEN_RETRIES = int(retries)

    # Guardar overrides de geração p/ answer_with_cfg()
    st.session_state["_gen_overrides"] = dict(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        timeout_s=timeout_s,
        retries=retries,
    )

def call_rag(question: str) -> tuple[str, list[str]]:
    """
    Usa o pipeline para responder e retorna também a lista de URLs (fontes) detectadas.
    """
    # Carregar uma vez o índice/mapping
    try:
        index, mapping = cached_load_search()
        ctx = search(index, mapping, question, k=cfg.TOP_K)
        urls = unique_urls_in_order(ctx, limit=5)
    except Exception as e:
        urls = []
        st.warning(f"Falha ao recuperar fontes: {e}")

    # Responder passando overrides de geração
    gen_over = st.session_state.get("_gen_overrides", {})
    answer_text = answer_with_cfg(question, gen_overrides=gen_over, k=cfg.TOP_K)

    return answer_text, urls

# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="Chatbot • RAG (LM Studio)", page_icon="🤖", layout="wide")
st.title("🤖 Chatbot de Transformação Digital (RAG + LM Studio)")

configure_runtime()

# Histórico
if "messages" not in st.session_state:
    st.session_state["messages"] = []  # cada item: {"role": "...", "text": str, "urls": [str]}

col_input, col_btn = st.columns([1, 0.15])
with col_input:
    question = st.text_input("Pergunte em PT-BR:", placeholder="Ex.: O que é transformação digital?")
with col_btn:
    submit = st.button("Enviar", type="primary", use_container_width=True)

st.divider()

# Render do histórico
for msg in st.session_state["messages"]:
    if msg["role"] == "user":
        st.chat_message("user").write(msg["text"])
    else:
        m = st.chat_message("assistant")
        m.write(msg["text"])
        if msg.get("urls"):
            with st.expander("Fontes (links)"):
                for u in msg["urls"]:
                    st.markdown(f"- {u}")

# Ação ao enviar
if submit and question.strip():
    st.session_state["messages"].append({"role": "user", "text": question, "urls": []})
    st.chat_message("user").write(question)

    with st.spinner("Consultando base e gerando resposta..."):
        try:
            answer_text, urls = call_rag(question)
        except Exception as e:
            answer_text, urls = f"Erro ao responder: {e}", []

    a = st.chat_message("assistant")
    a.write(answer_text)
    if urls:
        with st.expander("Fontes (links)"):
            for u in urls:
                st.markdown(f"- {u}")

    st.session_state["messages"].append({"role": "assistant", "text": answer_text, "urls": urls})

# Sidebar extra
st.sidebar.markdown("---")
st.sidebar.caption(f"Índice: `{cfg.FAISS_INDEX.name}` • SBERT: `{cfg.SBERT_MODEL}`")
st.sidebar.caption(f"LLM: `{cfg.LMSTUDIO_MODEL}` @ {cfg.LMSTUDIO_HOST}")