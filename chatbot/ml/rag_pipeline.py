from __future__ import annotations

from pathlib import Path
import re
import sqlite3
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
    ANSWER_SENTINEL,
    CONTINUE_MAX_ROUNDS,
    MIN_GENERATION_CHARS,
    KEYWORD_MIN_LEN,
    KEYWORD_MIN_HITS,
    MIN_ANSWER_COVERAGE,
    HYBRID_LAMBDA,
    # novos
    DOMAIN_GUARD_STRICT,
    USER_UPLOAD_BYPASS_GUARD,
    MIN_ANSWER_COVERAGE_USER,
    MIN_GENERATION_CHARS_USER,
)
from .llm_backends import LMStudioBackend, GenerationConfig

NOT_FOUND_TEXT = "Não encontrei essa informação na base de conhecimento."

# ====== System prompts (dinâmicos) ======
SYS_PT_DOMAIN = (
    "Você é um assistente especializado em transformação digital no setor público do Brasil. "
    "Responda SEMPRE em português do Brasil (pt-BR). "
    "Use APENAS o conteúdo do CONTEXTO fornecido. "
    f"Se o contexto não suportar, responda exatamente: '{NOT_FOUND_TEXT}'. "
    "Formato da resposta (só inclua se houver base no CONTEXTO):\n"
    "1) Definição objetiva em 2–3 frases.\n"
    "2) Pontos-chave em bullets (3–6 itens).\n"
    "3) Se houver no CONTEXTO: exemplos/indicadores aplicados ao setor público.\n"
    "4) Em resumo: 1 frase iniciando com 'Em resumo,'\n"
    f"Finalize com {ANSWER_SENTINEL} e nada após ele."
)

# quando o contexto é de PDF enviado pelo usuário → tema livre
SYS_PT_USER = (
    "Você é um assistente. Responda SEMPRE em português do Brasil (pt-BR). "
    "Use APENAS o conteúdo do CONTEXTO fornecido (não use conhecimento externo). "
    f"Se o contexto não suportar, responda exatamente: '{NOT_FOUND_TEXT}'. "
    "Formato da resposta:\n"
    "1) Responda de forma direta em 2–3 frases.\n"
    "2) Pontos-chave em bullets (3–6 itens) quando fizer sentido.\n"
    "3) Em resumo: 1 frase final iniciando com 'Em resumo,'\n"
    f"Finalize com {ANSWER_SENTINEL} e nada após ele."
)

def _system_prompt(use_user_mode: bool) -> str:
    return SYS_PT_USER if use_user_mode else SYS_PT_DOMAIN

# -------------------- Índice e mapping --------------------
def load_search():
    assert Path(FAISS_INDEX).exists(), f"FAISS não encontrado: {FAISS_INDEX}"
    assert Path(MAPPING_PARQUET).exists(), f"Mapping não encontrado: {MAPPING_PARQUET}"
    index = faiss.read_index(str(FAISS_INDEX))
    mapping = pd.read_parquet(MAPPING_PARQUET)
    return index, mapping

# -------------------- Embeddings --------------------
@lru_cache(maxsize=1)
def _get_sbert():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(SBERT_MODEL)

def _encode(texts: list[str]) -> np.ndarray:
    vec = _get_sbert().encode(texts, normalize_embeddings=True)
    return np.asarray(vec, dtype="float32")

# -------------------- Léxico / Híbrido --------------------
STOPWORDS = {
    "o","a","os","as","de","da","do","das","dos","e","é","em","no","na","nos","nas","um","uma",
    "para","por","com","sem","sobre","ao","à","aos","às","que","se","sua","seu","suas","seus",
    "como","qual","quais","quando","onde","porque","porquê","me","minha","minhas","te","são",
    "ser","estar","foi","já","mais","menos","muito","pouco","entre","até","desde"
}
def _normalize(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def _words(s: str) -> list[str]:
    return re.findall(r"[a-z0-9á-úà-ũâ-ûã-õç]+", _normalize(s))

def _keywords(s: str, min_len: int = KEYWORD_MIN_LEN) -> list[str]:
    return [w for w in _words(s) if w not in STOPWORDS and len(w) >= min_len]

def _lex_overlap(query: str, text: str) -> float:
    qk = set(_keywords(query))
    if not qk:
        return 0.0
    t_norm = _normalize(text)
    hits = sum(1 for w in qk if w in t_norm)
    return hits / max(1, len(qk))

def search(index, mapping: pd.DataFrame, query_text: str, k: int = TOP_K) -> pd.DataFrame:
    q = _encode([query_text])
    D, I = index.search(q, max(k, 10))
    hits = mapping.iloc[I[0]].copy()
    hits["score_vec"] = D[0].astype(float)

    hits = hits[hits["score_vec"] >= float(SCORE_CUTOFF)]
    if hits.empty:
        return hits.reset_index(drop=True)

    smin, smax = float(hits["score_vec"].min()), float(hits["score_vec"].max())
    denom = (smax - smin) or 1.0
    hits["score_vec_n"] = (hits["score_vec"] - smin) / denom
    hits["score_lex"] = [_lex_overlap(query_text, t) for t in hits["text"].astype(str)]
    lam = float(HYBRID_LAMBDA)
    hits["score_hybrid"] = (1.0 - lam) * hits["score_vec_n"] + lam * hits["score_lex"]
    hits = (
        hits.sort_values("score_hybrid", ascending=False)
            .drop_duplicates(subset=["url"], keep="first")
            .head(k)
            .reset_index(drop=True)
    )
    return hits

# -------------------- Detectar URLs de uploads do usuário --------------------
def _user_source_urls(urls: list[str]) -> set[str]:
    """
    Consulta o DB de scraping e retorna o subconjunto de URLs que pertencem a user_sources.
    Usa o mesmo caminho que a UI usa para rotular PDFs do usuário.
    """
    try:
        from chatbot.extraction.scraping import DB_PATH  # mesmo ponto usado na UI
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        placeholders = ",".join(["?"] * len(urls))
        rows = conn.execute(f"""
            SELECT d.url
            FROM user_sources us
            JOIN documents d ON d.id = us.doc_id
            WHERE d.url IN ({placeholders})
        """, urls).fetchall()
        conn.close()
        return {r["url"] for r in rows}
    except Exception:
        return set()

# -------------------- Prompt helpers --------------------
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_THINK_OPEN  = re.compile(r"<think>.*", re.DOTALL | re.IGNORECASE)
def _strip_think(s: str) -> str:
    s = _THINK_BLOCK.sub("", s)
    s = _THINK_OPEN.sub("", s)
    return s.strip()

def _strip_meta_prefixes(text: str) -> str:
    t = text.lstrip()
    t = re.sub(r"^(responda em pt[- ]?br\.?\s*)", "", t, flags=re.I)
    t = re.sub(r"^(responda em português.*?\s*)", "", t, flags=re.I)
    t = re.sub(r"^(instruções:\s*)", "", t, flags=re.I)
    return t.lstrip()

def _estimate_tokens(question: str, blocks: list[str]) -> int:
    in_chars = len(question) + sum(len(b) for b in blocks)
    want = int(in_chars / 3.5) + 400
    return max(350, min(GEN_MAX_TOKENS, want))

def _build_prompt(question: str, contexts: pd.DataFrame) -> tuple[str, list[str], list[str]]:
    ctx = contexts.head(5).reset_index(drop=True)
    urls, blocks = [], []
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
        f"- Se o contexto não suportar, diga: '{NOT_FOUND_TEXT}'.\n"
        "- Não cite fontes dentro do corpo.\n"
        f"- Finalize a resposta com {ANSWER_SENTINEL}.\n"
    )
    return user_prompt, urls, blocks

def _needs_continuation_dyn(text: str, min_len: int) -> bool:
    # Continua enquanto NÃO houver sentinela OU enquanto não atingiu o mínimo exigido
    if len(text) < min_len:
        return True
    return (ANSWER_SENTINEL not in text) and (not text.rstrip().endswith((".", "!", "?", ".”", "”")))

def _strip_sentinel(text: str) -> str:
    return text.replace(ANSWER_SENTINEL, "").strip()

def _context_is_relevant(question: str, blocks: list[str]) -> bool:
    q_tokens_all = [t for t in _words(question) if t not in STOPWORDS]
    q_keys = [t for t in q_tokens_all if len(t) >= KEYWORD_MIN_LEN]
    if not q_keys:
        return False
    ctx_text = _normalize(" ".join(blocks))
    key_hits = sum(1 for t in set(q_keys) if t in ctx_text)
    def _ngrams(tokens: list[str], n: int) -> list[str]:
        return [" ".join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
    phrases = _ngrams(q_tokens_all, 3) + _ngrams(q_tokens_all, 2)
    phrases = [p for p in phrases if len(p.replace(" ", "")) >= (2 * KEYWORD_MIN_LEN - 2)]
    phrase_hit = any(p in ctx_text for p in phrases)
    return key_hits >= KEYWORD_MIN_HITS or phrase_hit

def _answer_covered_by_context(answer: str, blocks: list[str], min_cov: float) -> bool:
    ans_keys = set(_keywords(answer))
    if not ans_keys:
        return False
    ctx_text = _normalize(" ".join(blocks))
    covered = sum(1 for t in ans_keys if t in ctx_text)
    return (covered / max(1, len(ans_keys))) >= min_cov

def _is_degenerate_yesno(s: str) -> bool:
    t = (s or "").strip().lower()
    t = re.sub(r"[^a-zá-úà-ũâ-ûã-õç]", "", t)
    return t in {"sim", "nao", "não"}

# -------------------- Geração com LLM --------------------
def answer_with_cfg(
    question: str,
    gen_overrides: dict | None = None,
    k: int | None = None,
    handles: tuple | None = None,
) -> tuple[str, list[str], str]:
    """
    Retorna: (texto_final_ou_NOT_FOUND, urls, debug_reason)
    """
    debug_reason = ""
    k = int(k or TOP_K)
    if handles is None:
        index, mapping = load_search()
    else:
        index, mapping = handles

    # Busca
    ctx = search(index, mapping, question, k=k)
    if ctx is None or ctx.empty:
        return NOT_FOUND_TEXT, [], "Gate1: nenhum contexto acima do cutoff."

    # Descobrir se há fontes do usuário
    all_urls = [str(u) for u in ctx["url"].fillna("").tolist()]
    user_urls = _user_source_urls([u for u in all_urls if u])
    has_user_ctx = bool(user_urls)

    # Se houver fontes do usuário, restringe o contexto só a elas
    if has_user_ctx and USER_UPLOAD_BYPASS_GUARD:
        ctx_user = ctx[ctx["url"].isin(user_urls)].reset_index(drop=True)

        if not ctx_user.empty:
            ctx = ctx_user

    # Prompt + URLs + blocos
    user_prompt, urls, blocks = _build_prompt(question, ctx)

    # Seleciona o system prompt adequado
    use_user_mode = bool(has_user_ctx and USER_UPLOAD_BYPASS_GUARD)
    system_prompt = _system_prompt(use_user_mode or not DOMAIN_GUARD_STRICT)

    # Mínimo dinâmico de caracteres
    min_len_required = (MIN_GENERATION_CHARS_USER if use_user_mode else MIN_GENERATION_CHARS)

    # Gate 1.5 — só aplica quando NÃO estamos em modo user-upload
    if not use_user_mode:
        if not _context_is_relevant(question, blocks):
            return NOT_FOUND_TEXT, urls, "Gate1.5: contexto recuperado não é lexicalmente relevante para a pergunta."

    # Config de geração
    need_tokens = _estimate_tokens(question, blocks)
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

    llm = LMStudioBackend(model=LMSTUDIO_MODEL, host=LMSTUDIO_HOST)
    try:
        llm.warm_up(timeout_s=40)
    except AttributeError:
        pass

    # Geração
    try:
        text = llm.generate(user_prompt, system=system_prompt, cfg=cfg)
    except Exception as e:
        return NOT_FOUND_TEXT, urls, f"LLM error: {e}"

    text = _strip_think(text).strip()
    text = _strip_meta_prefixes(text)

    # Degeneração SIM/NÃO
    if _is_degenerate_yesno(text):
        user_prompt += (
            "\n\nProduza a resposta COMPLETA conforme o formato, não responda apenas "
            "'SIM' ou 'NÃO'. Utilize bullets e finalize com o sentinela."
        )
        try:
            text = llm.generate(user_prompt, system=system_prompt, cfg=cfg)
        except Exception as e:
            return NOT_FOUND_TEXT, urls, f"LLM retry error: {e}"
        text = _strip_think(text).strip()
        text = _strip_meta_prefixes(text)

    history_text = text
    rounds = 0

    def _continuation_prefix() -> str:
        return (
            f"{user_prompt}\n\n"
            f"Texto já gerado (não repita):\n{history_text}\n\n"
            "CONTINUE do ponto em que parou, mantendo o mesmo tom e formato, "
            "usando SOMENTE o contexto acima. "
            f"Finalize com {ANSWER_SENTINEL} e nada após."
        )

    # >>> usa o verificador dinâmico para continuar até atingir o mínimo
    while _needs_continuation_dyn(history_text, min_len_required) and rounds < CONTINUE_MAX_ROUNDS:
        rounds += 1
        follow = _continuation_prefix()
        try:
            extra = llm.generate(follow, system=system_prompt, cfg=cfg)
        except Exception:
            break
        extra = _strip_think(extra).strip()
        extra = _strip_meta_prefixes(extra)
        if extra and extra not in history_text:
            if extra.startswith(history_text[:200]):
                extra = extra[len(history_text):].lstrip()
            history_text = (history_text + "\n\n" + extra).strip()
        else:
            break

    final_text = _strip_sentinel(history_text).strip()

    # Gate 3 — cobertura (limiar depende do modo) e comprimento dinâmico
    cov_min = MIN_ANSWER_COVERAGE_USER if use_user_mode else MIN_ANSWER_COVERAGE
    if not final_text or len(final_text) < min_len_required:
        return NOT_FOUND_TEXT, urls, f"Gate3: resposta curta/ausente. min={min_len_required}"
    if not _answer_covered_by_context(final_text, blocks, cov_min):
        top_hybrid = float(ctx.iloc[0].get("score_hybrid", 0.0))
        top_vec = float(ctx.iloc[0].get("score_vec", 0.0))
        top_lex = float(ctx.iloc[0].get("score_lex", 0.0))
        return NOT_FOUND_TEXT, urls, (
            f"Gate3: resposta não coberta pelo contexto. "
            f"top_hybrid={top_hybrid:.3f} top_vec={top_vec:.3f} top_lex={top_lex:.3f} "
            f"min_coverage={cov_min:.2f} (user_mode={use_user_mode})"
        )

    if final_text.strip() == NOT_FOUND_TEXT:
        return NOT_FOUND_TEXT, urls, "Modelo devolveu sentinel de não encontrado."

    return final_text, urls, ""
