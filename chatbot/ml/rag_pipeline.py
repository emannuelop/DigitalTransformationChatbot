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
    "Use APENAS o conteúdo do CONTEXTO fornecido (documentos/PDFs indexados). "
    "Se o contexto não suportar, responda exatamente: '{NOT_FOUND_TEXT}'. "

    "Regras gerais:\n"
    "- Não invente dados, números, leis, nomes de programas ou exemplos. Se não houver no CONTEXTO, diga: '{NOT_FOUND_TEXT}'.\n"
    "- Não use conhecimento externo nem opinião própria.\n"
    "- Seja específico, evite redundâncias e revise ortografia/acentuação.\n"

    "Formato da resposta (só inclua se houver base no CONTEXTO):\n"
    "1) Definição objetiva em 2–3 frases.\n"
    "2) Pontos-chave em bullets (3–6 itens), cada bullet com 1 ação/insight claro.\n"
    "3) Exemplos ou indicadores aplicados ao setor público, quando estiverem no CONTEXTO.\n"
    "4) Em resumo: 1 frase iniciando com 'Em resumo,'\n"

    "Na última linha, escreva SOMENTE o sentinela: {ANSWER_SENTINEL}"
)

# quando o contexto é de PDF enviado pelo usuário → tema livre
SYS_PT_USER = (
    "Você é um assistente. Responda SEMPRE em português do Brasil (pt-BR). "
    "Use APENAS o conteúdo do CONTEXTO fornecido (não use conhecimento externo). "
    f"Se o contexto não suportar, responda exatamente: '{NOT_FOUND_TEXT}'. "
    "Formato da resposta:\n"
    "1) Responda de forma direta e completa.\n"
    "2) Use bullets quando apropriado para organizar informações.\n"
    "3) Seja claro e objetivo.\n"
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

# --- SUBSTITUIR A FUNÇÃO search(...) ---
def search(index, mapping: pd.DataFrame, query_text: str, user_id: int | None = None, k: int = TOP_K) -> pd.DataFrame:
    q = _encode([query_text])
    D, I = index.search(q, max(k, 100))

    faiss_results = pd.DataFrame({'faiss_id': I[0], 'score_vec': D[0]})
    hits = faiss_results.merge(mapping, left_on='faiss_id', right_index=True, how='left')

    # 1) Filtro base por usuário:
    if user_id is not None:
        # documentos globais (user_id NULL) OU do próprio usuário
        hits = hits[hits["user_id"].isna() | (hits["user_id"] == user_id)]
    else:
        hits = hits[hits["user_id"].isna()]

    if hits.empty:
        return hits.reset_index(drop=True)

    # 2) Filtro anti-vazamento (se URL foi reclamado por QUALQUER usuário
    #    e não é do usuário atual, não pode aparecer como "global"):
    urls_in_hits = [str(u) for u in hits["url"].fillna("").tolist() if u]
    claimed_any = _urls_claimed_by_any_user(urls_in_hits)
    mine_urls = _user_source_urls(urls_in_hits, user_id) if user_id is not None else set()
    forbidden_globals = claimed_any - mine_urls

    if forbidden_globals:
        hits = hits[~(hits["user_id"].isna() & hits["url"].isin(forbidden_globals))]

    if hits.empty:
        return hits.reset_index(drop=True)

    # 3) Scoring e corte
    hits = hits.sort_values("score_vec", ascending=False).head(k).copy()
    hits["score_vec"] = hits["score_vec"].astype(float)
    hits = hits[hits["score_vec"] >= float(SCORE_CUTOFF)]
    if hits.empty:
        return hits.reset_index(drop=True)

    # 4) Híbrido
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
# --- SUBSTITUIR ESTA FUNÇÃO ---
def _user_source_urls(urls: list[str], user_id: int | None = None) -> set[str]:
    """
    Retorna o subconjunto de URLs (dentro de `urls`) que pertencem ao usuário `user_id`
    segundo a tabela user_sources. Usa o vínculo us.user_id (correto para posse).
    """
    if not urls or user_id is None:
        return set()

    try:
        from chatbot.extraction.scraping import DB_PATH
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row

        placeholders = ",".join(["?"] * len(urls))
        rows = conn.execute(
            f"""
            SELECT d.url
            FROM user_sources us
            JOIN documents d ON d.id = us.doc_id
            WHERE d.url IN ({placeholders})
              AND us.user_id = ?
            """,
            urls + [user_id],
        ).fetchall()
        conn.close()
        return {r["url"] for r in rows}
    except Exception as e:
        import sys
        print(f"[DEBUG] Erro em _user_source_urls: {e}", file=sys.stderr)
        return set()
    
# --- ADICIONAR ESTA NOVA FUNÇÃO ---
def _urls_claimed_by_any_user(urls: list[str]) -> set[str]:
    """
    Retorna o subconjunto de URLs (dentro de `urls`) que constam em user_sources para QUALQUER usuário.
    Serve para impedir que um URL 'global' (user_id = NULL no Parquet) vaze se ele já foi
    anexado por algum usuário específico.
    """
    if not urls:
        return set()

    try:
        from chatbot.extraction.scraping import DB_PATH
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row

        placeholders = ",".join(["?"] * len(urls))
        rows = conn.execute(
            f"""
            SELECT DISTINCT d.url
            FROM user_sources us
            JOIN documents d ON d.id = us.doc_id
            WHERE d.url IN ({placeholders})
            """,
            urls,
        ).fetchall()
        conn.close()
        return {r["url"] for r in rows}
    except Exception as e:
        import sys
        print(f"[DEBUG] Erro em _urls_claimed_by_any_user: {e}", file=sys.stderr)
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
    # Remove prefixos de continuação que o modelo gera incorretamente
    t = re.sub(r"^(continuando a resposta.*?:\s*)", "", t, flags=re.I)
    t = re.sub(r"^(continuando.*?:\s*)", "", t, flags=re.I)
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
    """
    Critério mais rigoroso para evitar loops desnecessários.
    Só continua se:
    1. Não tem a sentinela E
    2. Não termina com pontuação adequada E
    3. Está MUITO abaixo do mínimo (< 70% do mínimo)
    """
    # Se tem sentinela, nunca continua
    if ANSWER_SENTINEL in text:
        return False
    
    # Se termina com pontuação adequada e tem pelo menos 70% do mínimo, não continua
    if text.rstrip().endswith((".", "!", "?", "."", """)):
        if len(text) >= (min_len * 0.7):
            return False
    
    # Só continua se está muito abaixo do mínimo
    return len(text) < (min_len * 0.7)

def _strip_sentinel(text: str) -> str:
    return text.replace(ANSWER_SENTINEL, "").strip()

def _context_is_relevant(question: str, blocks: list[str]) -> bool:
    """
    Verifica se o contexto recuperado é lexicalmente relevante para a pergunta.
    """
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
    """
    Verifica cobertura de forma mais permissiva.
    Aceita palavras parciais e variações.
    """
    ans_keys = set(_keywords(answer, min_len=4))  # Reduz min_len de 5 para 4
    if not ans_keys:
        return True  # Se não há keywords, aceita (evita rejeição por falta de palavras-chave)
    
    ctx_text = _normalize(" ".join(blocks))
    
    # Conta matches exatos
    covered_exact = sum(1 for t in ans_keys if t in ctx_text)
    
    # Conta matches parciais (substring)
    covered_partial = sum(1 for t in ans_keys if any(t in word or word in t for word in ctx_text.split()))
    
    # Usa o maior dos dois
    covered = max(covered_exact, covered_partial)
    
    coverage = covered / max(1, len(ans_keys))
    return coverage >= min_cov

def _is_degenerate_yesno(s: str) -> bool:
    t = (s or "").strip().lower()
    t = re.sub(r"[^a-zá-úà-ũâ-ûã-õç]", "", t)
    return t in {"sim", "nao", "não"}

# -------------------- Geração com LLM --------------------
def answer_with_cfg(
    question: str,
    user_id: int | None = None,
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
    ctx = search(index, mapping, question, user_id=user_id, k=k)
    if ctx is None or ctx.empty:
        return NOT_FOUND_TEXT, [], "Gate1: nenhum contexto acima do cutoff."

    # Descobrir se há fontes do usuário DESTE USUÁRIO ESPECÍFICO
    all_urls = [str(u) for u in ctx["url"].fillna("").tolist()]
    # CORRIGIDO: Passa user_id para filtrar corretamente
    user_urls = _user_source_urls([u for u in all_urls if u], user_id=user_id)
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

# --- SUBSTITUIR O TRECHO "Gate 1.5" DENTRO DE answer_with_cfg(...) POR ESTE ---
    # NOVO Gate 1.5 — relevância obrigatória no modo global (anti 'Minecraft')
    if not use_user_mode:
        top_score = float(ctx.iloc[0].get("score_hybrid", 0.0)) if not ctx.empty else 0.0
        is_relevant = _context_is_relevant(question, blocks)

        # Se a base recuperada não tem relevância lexical suficiente, corta logo.
        if not is_relevant:
            return NOT_FOUND_TEXT, [], f"Gate1.5: contexto sem relevância lexical (score={top_score:.3f})"

        # Se o score for MUITO baixo, corta também.
        if top_score < 0.25:
            return NOT_FOUND_TEXT, [], f"Gate1.5: score muito baixo ({top_score:.3f})"

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

    # Lógica de continuação mais conservadora
    def _continuation_prefix() -> str:
        return (
            f"Pergunta: {question}\n\n"
            f"Contexto:\n{chr(10).join(blocks)}\n\n"
            f"Resposta parcial já gerada:\n{history_text}\n\n"
            "Complete a resposta acima de forma natural, adicionando informações relevantes do contexto. "
            "NÃO repita o que já foi dito. "
            f"Finalize com {ANSWER_SENTINEL}."
        )

    # Usa o verificador dinâmico para continuar com critério mais rigoroso
    while _needs_continuation_dyn(history_text, min_len_required) and rounds < CONTINUE_MAX_ROUNDS:
        rounds += 1
        follow = _continuation_prefix()
        try:
            extra = llm.generate(follow, system=system_prompt, cfg=cfg)
        except Exception:
            break
        extra = _strip_think(extra).strip()
        extra = _strip_meta_prefixes(extra)
        
        # Verifica se a continuação é válida (não é repetição)
        if not extra or extra in history_text:
            break
        
        # Verifica se começou a repetir
        if extra.startswith(history_text[:100]):
            break
            
        # Verifica se tem o padrão de repetição problemático
        if "continuando a resposta" in extra.lower():
            break
        
        if extra and extra not in history_text:
            if extra.startswith(history_text[:200]):
                extra = extra[len(history_text):].lstrip()
            history_text = (history_text + "\n\n" + extra).strip()
        else:
            break

    final_text = _strip_sentinel(history_text).strip()

    # Gate 3 — cobertura (limiar depende do modo) e comprimento dinâmico
    cov_min = MIN_ANSWER_COVERAGE_USER if use_user_mode else MIN_ANSWER_COVERAGE
    
    # Verifica comprimento mínimo
    if not final_text or len(final_text) < (min_len_required * 0.5):
        return NOT_FOUND_TEXT, urls, f"Gate3: resposta curta/ausente. min={min_len_required}, got={len(final_text)}"
    
    # Gate de cobertura mais permissivo para user_mode
    if use_user_mode:
        # Para PDFs do usuário, apenas verifica se há ALGUMA cobertura
        if not _answer_covered_by_context(final_text, blocks, cov_min):
            # Log mas não rejeita automaticamente
            debug_reason = f"Baixa cobertura em user_mode (cov_min={cov_min}), mas aceitando resposta."
    else:
        # Para conteúdo global, mantém verificação mais rigorosa
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

    return final_text, urls, debug_reason
