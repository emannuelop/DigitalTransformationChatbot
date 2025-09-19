# chatbot/ml/rag_pipeline.py
from pathlib import Path
import re
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

from .settings import (
    FAISS_INDEX, MAPPING_PARQUET, TOP_K, MAX_CHARS_PER_CHUNK,
    LMSTUDIO_HOST, LMSTUDIO_MODEL, SBERT_MODEL
)
from .llm_backends import LMStudioBackend, GenerationConfig

# Carrega SBERT UMA vez
_SBERT = SentenceTransformer(SBERT_MODEL)

SYS_PT = (
    "Você é um assistente especializado em transformação digital no setor público do Brasil. "
    "Responda SEMPRE em português do Brasil (pt-BR). Baseie-se apenas no CONTEXTO abaixo. "
    "Se não houver base suficiente, diga exatamente: 'Não encontrei essa informação na base de conhecimento.' "
    "Escreva um ou mais parágrafos coesos. Não cite fontes dentro do texto; os links virão depois."
)

def load_search():
    assert Path(FAISS_INDEX).exists(), f"FAISS não encontrado: {FAISS_INDEX}"
    assert Path(MAPPING_PARQUET).exists(), f"Mapping não encontrado: {MAPPING_PARQUET}"
    index = faiss.read_index(str(FAISS_INDEX))
    mapping = pd.read_parquet(MAPPING_PARQUET)
    return index, mapping

def search(index, mapping: pd.DataFrame, query_text: str, k: int = TOP_K):
    q = _SBERT.encode([query_text], normalize_embeddings=True)
    D, I = index.search(np.array(q, dtype="float32"), k)
    hits = mapping.iloc[I[0]].copy()
    hits["score"] = D[0]
    return hits

def build_prompt(question: str, contexts: pd.DataFrame):
    ctx = contexts.drop_duplicates(subset=["url"], keep="first").head(5).reset_index(drop=True)

    urls, blocks = [], []
    for _, r in ctx.iterrows():
        url = (r.get("url") or "").strip() or "fonte desconhecida"
        if url not in urls and len(urls) < 5:
            urls.append(url)
        snippet = (r.get("text") or "")[:MAX_CHARS_PER_CHUNK]
        blocks.append(f"- {url}\n{snippet}")

    context_txt = "\n\n".join(blocks)
    user_prompt = (
        f"Pergunta: {question}\n\n"
        f"Contexto (use SOMENTE isso para responder):\n{context_txt}\n\n"
        "Responda em pt-BR, com um ou mais parágrafos, sem citar fontes no corpo."
    )
    return user_prompt, urls

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_THINK_OPEN  = re.compile(r"<think>.*", re.DOTALL | re.IGNORECASE)

def _strip_think(text: str) -> str:
    text = _THINK_BLOCK.sub("", text)
    text = _THINK_OPEN.sub("", text)
    return text.strip()

def _pt_br_cleanup(text: str) -> str:
    repl = {
        r"\bcompetitive advantage\b": "vantagem competitiva",
        r"\bdigitales\b": "digitais",
        r"\binvolucra\b": "envolve",
    }
    for a, b in repl.items():
        text = re.sub(a, b, text, flags=re.IGNORECASE)
    text = re.sub(r"\s+\.", ".", text)
    text = re.sub(r"([.!?])\s+([a-z])", lambda m: f"{m.group(1)} {m.group(2).upper()}", text)
    return text.strip()

def answer(question: str, k: int = TOP_K) -> str:
    index, mapping = load_search()
    ctx = search(index, mapping, question, k=k)
    user_prompt, urls = build_prompt(question, ctx)

    llm = LMStudioBackend(model=LMSTUDIO_MODEL, host=LMSTUDIO_HOST)
    cfg = GenerationConfig(temperature=0.2, top_p=0.9, max_tokens=1200, timeout_s=300)

    text = llm.generate(user_prompt, system=SYS_PT, cfg=cfg)
    text = _pt_br_cleanup(_strip_think(text))

    if not text:
        text = "Não encontrei essa informação na base de conhecimento."

    out = ["Resposta:", text, "", "Fontes (links):"]
    out += [f"- {u}" for u in urls] if urls else ["(nenhuma fonte disponível)"]
    return "\n".join(out)

def answer_with_cfg(question: str, gen_overrides: dict | None = None, k: int = TOP_K) -> str:
    """
    Igual ao answer(), mas aceita overrides de geração:
      gen_overrides = {"temperature": float, "top_p": float, "max_tokens": int}
    """
    index, mapping = load_search()
    ctx = search(index, mapping, question, k=k)
    user_prompt, urls = build_prompt(question, ctx)

    llm = LMStudioBackend(model=LMSTUDIO_MODEL, host=LMSTUDIO_HOST)

    # Defaults
    cfg = GenerationConfig(
        temperature=0.2,
        top_p=0.9,
        max_tokens=1200,
        timeout_s=300,
        stop=None,
    )
    # Overrides vindos da UI (se existirem)
    if gen_overrides:
        if "temperature" in gen_overrides: cfg.temperature = float(gen_overrides["temperature"])
        if "top_p" in gen_overrides:       cfg.top_p       = float(gen_overrides["top_p"])
        if "max_tokens" in gen_overrides:  cfg.max_tokens  = int(gen_overrides["max_tokens"])

    # Primeira tentativa
    text = llm.generate(user_prompt, system=SYS_PT, cfg=cfg)
    text = _strip_think(text)
    text = _pt_br_cleanup(text)

    # Fallback
    if not text.strip():
        fallback_user = (
            user_prompt
            + "\n\nATENÇÃO: NÃO use <think> e NÃO mostre raciocínio."
            + " Forneça apenas a resposta final, em pt-BR, com um ou mais parágrafos."
        )
        text = llm.generate(fallback_user, system=SYS_PT, cfg=cfg)
        text = _strip_think(text)
        text = _pt_br_cleanup(text)

    if not text.strip():
        text = "Não encontrei essa informação na base de conhecimento."

    lines = ["Resposta:", text.strip(), "", "Fontes (links):"]
    if urls:
        for u in urls:
            lines.append(f"- {u}")
    else:
        lines.append("(nenhuma fonte disponível)")
    return "\n".join(lines)

if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or "O que é governo digital no Brasil?"
    print(answer(q))
