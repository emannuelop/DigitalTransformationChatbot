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
    FORCE_PT_BR,  # mantido por compatibilidade, mas não é usado para prefixar o prompt
    ANSWER_SENTINEL,
    CONTINUE_MAX_ROUNDS,
    MIN_GENERATION_CHARS,
    JUDGE_ANSWERABILITY,
    JUDGE_MAX_TOKENS,
    KEYWORD_MIN_LEN,
    KEYWORD_MIN_HITS,
    MIN_ANSWER_COVERAGE,
    HYBRID_LAMBDA,
)
from .llm_backends import LMStudioBackend, GenerationConfig

# Texto padrão para quando não há base suficiente
NOT_FOUND_TEXT = "Não encontrei essa informação na base de conhecimento."

# Mensagem de sistema (resposta estruturada, em pt-BR e focada no domínio)
SYS_PT = (
    "Você é um assistente especializado em transformação digital no setor público do Brasil. "
    "Responda SEMPRE em português do Brasil (pt-BR). "
    "Use APENAS o conteúdo do CONTEXTO fornecido. "
    "NUNCA inicie a resposta com meta-instruções (ex.: 'Responda em pt-BR.'). "
    "Foque no setor público brasileiro (serviços públicos, eficiência, transparência, "
    "experiência do cidadão, gestão e habilidades digitais) quando isso estiver no CONTEXTO. "
    f"Se o contexto não suportar, responda exatamente: '{NOT_FOUND_TEXT}'. "
    "Formato da resposta (só inclua se houver base no CONTEXTO):\n"
    "1) Definição objetiva em 2–3 frases, sem jargões.\n"
    "2) Pontos-chave em bullets (3–6 itens), cada bullet com 1 frase curta.\n"
    "3) Se houver no CONTEXTO: exemplos/indicadores aplicados ao setor público.\n"
    "4) Em resumo: 1 frase final iniciando com 'Em resumo,'\n"
    "Evite redundância e não repita o texto do contexto literalmente. "
    f"No final da resposta, escreva o marcador {ANSWER_SENTINEL} e nada após ele."
)

# ----------------------------------------------------------------------
# Carregamento do índice e mapping
# ----------------------------------------------------------------------
def load_search():
    """Lê o índice FAISS e o mapping parquet."""
    assert Path(FAISS_INDEX).exists(), f"FAISS não encontrado: {FAISS_INDEX}"
    assert Path(MAPPING_PARQUET).exists(), f"Mapping não encontrado: {MAPPING_PARQUET}"
    index = faiss.read_index(str(FAISS_INDEX))
    mapping = pd.read_parquet(MAPPING_PARQUET)
    return index, mapping

# ----------------------------------------------------------------------
# Embeddings (cache do modelo SBERT) e codificação
# ----------------------------------------------------------------------
@lru_cache(maxsize=1)
def _get_sbert():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(SBERT_MODEL)

def _encode(texts: list[str]) -> np.ndarray:
    vec = _get_sbert().encode(texts, normalize_embeddings=True)
    return np.asarray(vec, dtype="float32")

# ----------------- utilitários léxicos / hibridização -----------------
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
    """Score léxico simples: fração de palavras-chave do usuário presentes no chunk."""
    qk = set(_keywords(query))
    if not qk:
        return 0.0
    t_norm = _normalize(text)
    hits = sum(1 for w in qk if w in t_norm)
    return hits / max(1, len(qk))

def search(index, mapping: pd.DataFrame, query_text: str, k: int = TOP_K) -> pd.DataFrame:
    """Busca vetorial + reranking híbrido (FAISS + overlap léxico)."""
    q = _encode([query_text])
    D, I = index.search(q, max(k, 10))  # pega um pouco mais para reranqueamento
    hits = mapping.iloc[I[0]].copy()
    hits["score_vec"] = D[0].astype(float)

    # filtra por cutoff vetorial primeiro
    hits = hits[hits["score_vec"] >= float(SCORE_CUTOFF)]
    if hits.empty:
        return hits.reset_index(drop=True)

    # normaliza score vetorial para [0,1]
    smin, smax = float(hits["score_vec"].min()), float(hits["score_vec"].max())
    denom = (smax - smin) or 1.0
    hits["score_vec_n"] = (hits["score_vec"] - smin) / denom

    # componente léxica
    hits["score_lex"] = [_lex_overlap(query_text, t) for t in hits["text"].astype(str)]

    # híbrido
    lam = float(HYBRID_LAMBDA)
    hits["score_hybrid"] = (1.0 - lam) * hits["score_vec_n"] + lam * hits["score_lex"]

    # remove duplicados por URL e ordena pelo híbrido
    hits = (
        hits.sort_values("score_hybrid", ascending=False)
            .drop_duplicates(subset=["url"], keep="first")
            .head(k)
            .reset_index(drop=True)
    )
    return hits

# ----------------------------------------------------------------------
# Utilitários de prompt e limpeza
# ----------------------------------------------------------------------
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_THINK_OPEN  = re.compile(r"<think>.*", re.DOTALL | re.IGNORECASE)

def _strip_think(s: str) -> str:
    s = _THINK_BLOCK.sub("", s)
    s = _THINK_OPEN.sub("", s)
    return s.strip()

def _strip_meta_prefixes(text: str) -> str:
    """Remove linhas de meta-instruções que o modelo possa ecoar no início."""
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
        "- Siga o formato solicitado no sistema.\n"
        f"- Se o contexto não suportar, diga: '{NOT_FOUND_TEXT}'.\n"
        "- Não cite fontes dentro do corpo.\n"
        f"- Finalize a resposta com {ANSWER_SENTINEL}.\n"
    )
    return user_prompt, urls, blocks

def _needs_continuation(text: str) -> bool:
    """Continuar se não houver sentinela, estiver curto ou terminar truncado."""
    if ANSWER_SENTINEL in text:
        return False
    if len(text) < MIN_GENERATION_CHARS:
        return True
    return not text.rstrip().endswith((".", "!", "?", ".”", '”'))

def _strip_sentinel(text: str) -> str:
    return text.replace(ANSWER_SENTINEL, "").strip()

def _context_is_relevant(question: str, blocks: list[str]) -> bool:
    """Gate determinístico: precisa bater com palavras-chave/frases do usuário."""
    q_tokens_all = [t for t in _words(question) if t not in STOPWORDS]
    q_keys = [t for t in q_tokens_all if len(t) >= KEYWORD_MIN_LEN]
    if not q_keys:
        return False

    ctx_text = _normalize(" ".join(blocks))
    key_hits = sum(1 for t in set(q_keys) if t in ctx_text)

    # frases (bigrams/trigrams) — ajuda com termos compostos
    def _ngrams(tokens: list[str], n: int) -> list[str]:
        return [" ".join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
    phrases = _ngrams(q_tokens_all, 3) + _ngrams(q_tokens_all, 2)
    phrases = [p for p in phrases if len(p.replace(" ", "")) >= (2 * KEYWORD_MIN_LEN - 2)]
    phrase_hit = any(p in ctx_text for p in phrases)

    return key_hits >= KEYWORD_MIN_HITS or phrase_hit

def _answer_covered_by_context(answer: str, blocks: list[str]) -> bool:
    """Pós-geração: a resposta precisa ter cobertura léxica no contexto."""
    ans_keys = set(_keywords(answer))
    if not ans_keys:
        return False
    ctx_text = _normalize(" ".join(blocks))
    covered = sum(1 for t in ans_keys if t in ctx_text)
    return (covered / max(1, len(ans_keys))) >= MIN_ANSWER_COVERAGE

# ----------------------------------------------------------------------
# Geração com o LLM
# ----------------------------------------------------------------------
def answer_with_cfg(
    question: str,
    gen_overrides: dict | None = None,
    k: int | None = None,
    handles: tuple | None = None,
) -> tuple[str, list[str]]:
    """
    Responde com base em RAG + LMStudio.
    Retorna: (texto_final, lista_de_urls_usadas)
    """
    k = int(k or TOP_K)
    if handles is None:
        index, mapping = load_search()
    else:
        index, mapping = handles

    # Busca contexto
    ctx = search(index, mapping, question, k=k)

    # Gate 1: sem contexto útil
    if ctx is None or ctx.empty:
        return NOT_FOUND_TEXT, []

    # Prompt + URLs
    user_prompt, urls, blocks = _build_prompt(question, ctx)

    # Gate 1.5: relevância léxica do contexto vs. pergunta
    if not _context_is_relevant(question, blocks):
        return NOT_FOUND_TEXT, []

    # Heurística de tokens
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

    # NÃO prefixar "Responda em pt-BR." no prompt do usuário; isso fica no system.

    # Cliente LLM
    llm = LMStudioBackend(model=LMSTUDIO_MODEL, host=LMSTUDIO_HOST)
    try:
        llm.warm_up(timeout_s=40)
    except AttributeError:
        pass

    # Gate 2 (judge probabilístico)
    try:
        if JUDGE_ANSWERABILITY:
            judge_prompt = (
                "Você receberá uma pergunta e blocos de contexto. "
                "Responda SOMENTE com 'SIM' se o contexto permite responder com segurança; "
                "caso contrário, responda 'NÃO'. "
                "Não explique.\n\n"
                f"Pergunta: {question}\n\n"
                "Contexto:\n" + "\n".join(blocks[:3])
            )
            jcfg = GenerationConfig(temperature=0.0, top_p=1.0, max_tokens=JUDGE_MAX_TOKENS, timeout_s=60, retries=1)
            out = llm.generate(judge_prompt, system="Responda apenas SIM ou NÃO.", cfg=jcfg).strip().upper()
            if not out.startswith("SIM"):
                return NOT_FOUND_TEXT, []
    except Exception:
        # se o judge falhar, seguimos (os gates seguintes ainda protegem)
        pass

    # === Geração com sentinela + continuações ===
    history_text = ""
    rounds = 0

    def _continuation_prefix() -> str:
        return (
            f"{user_prompt}\n\n"
            f"Texto já gerado (não repita):\n{history_text}\n\n"
            "CONTINUE do ponto em que parou, mantendo o mesmo tom e formato, "
            "usando SOMENTE o contexto acima. "
            f"Finalize com {ANSWER_SENTINEL} e nada após."
        )

    try:
        text = llm.generate(user_prompt, system=SYS_PT, cfg=cfg)
    except Exception:
        return NOT_FOUND_TEXT, []

    text = _strip_think(text).strip()
    text = _strip_meta_prefixes(text)
    history_text = text

    while _needs_continuation(history_text) and rounds < CONTINUE_MAX_ROUNDS:
        rounds += 1
        follow = _continuation_prefix()
        try:
            extra = llm.generate(follow, system=SYS_PT, cfg=cfg)
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

    if not history_text:
        return NOT_FOUND_TEXT, []

    # Remove sentinela
    final_text = _strip_sentinel(history_text).strip()
    if not final_text:
        return NOT_FOUND_TEXT, []

    # Se o modelo escreveu explicitamente o texto de não encontrado, não mostrar fontes
    if final_text.strip() == NOT_FOUND_TEXT:
        return NOT_FOUND_TEXT, []

    # Gate 3: verificação pós-geração — cobertura da resposta pelo contexto
    if not _answer_covered_by_context(final_text, blocks):
        return NOT_FOUND_TEXT, []

    # Se passou em todos os gates, devolve resposta + URLs usadas
    return final_text, urls

# Back-compat
def answer(question: str, k: int | None = None) -> str:
    text, _ = answer_with_cfg(question, gen_overrides=None, k=k)
    return text

# CLI
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
