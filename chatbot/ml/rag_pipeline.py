# chatbot/ml/rag_pipeline.py
from __future__ import annotations

from pathlib import Path
import re
from functools import lru_cache

import numpy as np
import pandas as pd
import faiss

from .settings import (
    FAISS_INDEX,
    MAPPING_PARQUET,
    TOP_K,
    MAX_CHARS_PER_CHUNK,
    SCORE_CUTOFF,
    LMSTUDIO_HOST,
    LMSTUDIO_MODEL,
    SBERT_MODEL,
    GEN_TEMPERATURE,
    GEN_TOP_P,
    GEN_MAX_TOKENS,
    GEN_TIMEOUT_S,
    GEN_RETRIES,
    GEN_BACKOFF_S,
    GEN_STOP,
    FORCE_PT_BR,
)
from .llm_backends import LMStudioBackend, GenerationConfig


# Mensagem de sistema (instruções de alto nível para o LLM)
SYS_PT = (
    "Você é um assistente especializado em transformação digital no setor público do Brasil. "
    "Responda SEMPRE em português do Brasil (pt-BR). Baseie-se apenas no CONTEXTO. "
    "Se não houver base suficiente, responda exatamente: 'Não encontrei essa informação na base de conhecimento.' "
    "Não cite fontes dentro do texto."
)


# ------------------------------------------------------------------------------
# Carregamento do índice e mapping
# ------------------------------------------------------------------------------
def load_search():
    """Lê o índice FAISS e o mapping parquet."""
    assert Path(FAISS_INDEX).exists(), f"FAISS não encontrado: {FAISS_INDEX}"
    assert Path(MAPPING_PARQUET).exists(), f"Mapping não encontrado: {MAPPING_PARQUET}"
    index = faiss.read_index(str(FAISS_INDEX))
    mapping = pd.read_parquet(MAPPING_PARQUET)
    return index, mapping


# ------------------------------------------------------------------------------
# Embeddings (cache do modelo SBERT) e busca vetorial
# ------------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _get_sbert():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(SBERT_MODEL)

def _encode(texts: list[str]) -> np.ndarray:
    vec = _get_sbert().encode(texts, normalize_embeddings=True)
    return np.asarray(vec, dtype="float32")

def search(index, mapping: pd.DataFrame, query_text: str, k: int = TOP_K) -> pd.DataFrame:
    """Busca os k chunks mais similares e aplica deduplicação + filtro por score."""
    q = _encode([query_text])
    D, I = index.search(q, k)
    hits = mapping.iloc[I[0]].copy()
    hits["score"] = D[0]

    # 1) remove duplicados por URL mantendo a primeira ocorrência
    hits = hits.drop_duplicates(subset=["url"], keep="first")

    # 2) filtra por SCORE_CUTOFF se houver pelo menos 1 acima do limite
    filt = hits[hits["score"] >= float(SCORE_CUTOFF)]
    if len(filt) >= 1:
        hits = filt

    return hits.reset_index(drop=True)


# ------------------------------------------------------------------------------
# Utilitários de prompt e limpeza
# ------------------------------------------------------------------------------
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_THINK_OPEN  = re.compile(r"<think>.*", re.DOTALL | re.IGNORECASE)

def _strip_think(s: str) -> str:
    s = _THINK_BLOCK.sub("", s)
    s = _THINK_OPEN.sub("", s)
    return s.strip()

def _estimate_tokens(question: str, blocks: list[str]) -> int:
    """
    Estima tokens de saída a partir do tamanho do input, com limites.
    Piso: 256 para não ficar curto; Teto: GEN_MAX_TOKENS (settings).
    """
    in_chars = len(question) + sum(len(b) for b in blocks)
    want = int(in_chars / 3.5) + 300  # heurística um pouco mais “generosa”
    return max(256, min(GEN_MAX_TOKENS, want))

def _build_prompt(question: str, contexts: pd.DataFrame) -> tuple[str, list[str], list[str]]:
    """
    Monta o prompt final ao LLM.
    Retorna: (user_prompt, urls_usadas, blocos_de_contexto)
    """
    ctx = contexts.head(5).reset_index(drop=True)

    urls: list[str] = []
    blocks: list[str] = []
    for _, r in ctx.iterrows():
        url = (r.get("url") or "").strip() or "fonte desconhecida"
        if url not in urls and len(urls) < 5:
            urls.append(url)
        snippet = (r.get("text") or "")[:MAX_CHARS_PER_CHUNK]
        blocks.append(f"- Fonte: {url}\n{snippet}")

    context_txt = "\n\n".join(blocks)

    user_prompt = (
        f"Pergunta: {question}\n\n"
        f"Contexto (use SOMENTE o conteúdo abaixo):\n{context_txt}\n\n"
        "Instruções:\n"
        "- Responda em pt-BR, completos e coesos.\n"
        "- Se o contexto não suportar, diga: 'Não encontrei essa informação na base de conhecimento.'\n"
        "- Não repita trechos do contexto; integre-os em texto corrido.\n"
    )
    return user_prompt, urls, blocks


# ------------------------------------------------------------------------------
# Geração com o LLM (LM Studio)
# ------------------------------------------------------------------------------
def answer_with_cfg(
    question: str,
    gen_overrides: dict | None = None,
    k: int | None = None,
    handles: tuple | None = None,
) -> tuple[str, list[str]]:
    """
    Responde com base em RAG + LMStudio (OpenAI-like).
    Retorna: (texto_final, lista_de_urls_usadas)
    """
    k = int(k or TOP_K)
    if handles is None:
        index, mapping = load_search()
    else:
        index, mapping = handles

    # Busca contexto
    ctx = search(index, mapping, question, k=k)

    # 🚀 Atalho: sem contexto útil, nem chama o LLM
    if ctx is None or ctx.empty:
        return "Não encontrei essa informação na base de conhecimento.", []

    # Prompt + URLs para exibir na UI
    user_prompt, urls, blocks = _build_prompt(question, ctx)

    # Heurística de tokens de saída
    need_tokens = _estimate_tokens(question, blocks)

    # Configuração de geração (sobrescrevível pela UI)
    cfg = GenerationConfig(
        temperature=GEN_TEMPERATURE,
        top_p=GEN_TOP_P,
        max_tokens=need_tokens,
        timeout_s=GEN_TIMEOUT_S,
        retries=GEN_RETRIES,
        backoff_s=GEN_BACKOFF_S,
        stop=GEN_STOP,
    )
    if gen_overrides:
        for k_, v in gen_overrides.items():
            if hasattr(cfg, k_) and v is not None:
                setattr(cfg, k_, v)

    if FORCE_PT_BR:
        user_prompt = "Responda em pt-BR.\n\n" + user_prompt

    # Cliente LLM
    llm = LMStudioBackend(model=LMSTUDIO_MODEL, host=LMSTUDIO_HOST)
    # Alguns ambientes têm warm_up; se não tiver, simplesmente ignora
    try:
        llm.warm_up(timeout_s=40)  # cold-start mais estável
    except AttributeError:
        pass

    # === Primeira passada ===
    text = llm.generate(user_prompt, system=SYS_PT, cfg=cfg)
    text = _strip_think(text).strip()

    # === Continuação real se ficou curta ===
    # Enviamos o texto já gerado para o modelo CONTINUAR (sem repetir)
    if len(text) < 600:
        follow = (
            f"{user_prompt}\n\n"
            "Texto já gerado (parte 1):\n"
            f"{text}\n\n"
            "Agora CONTINUE a resposta do ponto em que parou, sem repetir o que já foi dito, "
            "mantendo o mesmo tom, coerência e em pt-BR."
        )
        text2 = llm.generate(follow, system=SYS_PT, cfg=cfg)
        text2 = _strip_think(text2).strip()
        if len(text2) > len(text):
            text = text2

    if not text:
        text = "Não encontrei essa informação na base de conhecimento."

    return text, urls


# ------------------------------------------------------------------------------
# Back-compat: função simples que retorna só o texto
# ------------------------------------------------------------------------------
def answer(question: str, k: int | None = None) -> str:
    text, _ = answer_with_cfg(question, gen_overrides=None, k=k)
    return text


# ------------------------------------------------------------------------------
# Execução via CLI (útil para teste rápido)
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or "O que é transformação digital?"
    txt, urls = answer_with_cfg(q)
    print("\nResposta:\n")
    print(txt)
    if urls:
        print("\nFontes (links):")
        for u in urls:
            print("-", u)
