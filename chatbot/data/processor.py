# processor.py
from __future__ import annotations

import argparse
import logging
import re
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from langdetect import DetectorFactory, detect
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from unidecode import unidecode

# --------------------------------------------------------------------------------------
# Configs/paths
# --------------------------------------------------------------------------------------
DATA_DIR = Path(__file__).resolve().parent             # chatbot/data
PROJECT_ROOT = DATA_DIR.parent                         # chatbot/
RAW_DB_PATH = PROJECT_ROOT / "crawler" / "data" / "knowledge_base.db"
PROCESSED_DB_PATH = DATA_DIR / "knowledge_base_processed.db"
ARTIFACTS_DIR = DATA_DIR / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# Reprodutibilidade da detecção de idioma
DetectorFactory.seed = 42

# Regex pré-compiladas
_RX_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
_RX_SPACES = re.compile(r"\s+")

# Categorias simples por keywords (extensível)
CATEGORIES: Dict[str, List[str]] = {
    "estrategia": ["estrategia", "roadmap", "planejamento", "governanca", "maturidade", "diretriz"],
    "servicos_digitais": ["servico digital", "gov.br", "portal", "experiencia do usuario", "ux", "acessibilidade"],
    "dados_ia": ["dados", "governanca de dados", "ia", "inteligencia artificial", "machine learning", "analytics", "algoritmo", "aprendizado"],
    "processos": ["processo", "automacao", "bpmn", "workflow", "otimizacao", "reengenharia"],
    "infra_seg": ["nuvem", "cloud", "seguranca", "lgpd", "privacidade", "api", "arquitetura"],
    "setor_publico": ["setor publico", "governo digital", "politica publica", "decreto", "licitacao"],
}

# --------------------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------------------
def setup_logging(level: int = logging.INFO) -> None:
    fmt = "[%(levelname)s] %(message)s"
    logging.basicConfig(level=level, format=fmt)


# --------------------------------------------------------------------------------------
# SBERT (lazy import)
# --------------------------------------------------------------------------------------
def load_sbert(model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
    """Carrega SentenceTransformer apenas quando necessário."""
    from sentence_transformers import SentenceTransformer  # lazy
    return SentenceTransformer(model_name)


# --------------------------------------------------------------------------------------
# DB helpers
# --------------------------------------------------------------------------------------
def ensure_raw_db_exists() -> None:
    if not RAW_DB_PATH.exists():
        raise FileNotFoundError(f"DB bruto não encontrado: {RAW_DB_PATH}. Rode o crawler antes.")


def init_processed_schema(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS processed_documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            original_id INTEGER UNIQUE,
            url TEXT,
            lang TEXT,
            category TEXT,
            content_clean TEXT,
            content_len INTEGER,
            text_hash TEXT,
            processed_at TEXT
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS duplicates_map (
            original_id INTEGER,
            kept_original_id INTEGER
        )
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_proc_orig ON processed_documents(original_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_proc_hash ON processed_documents(text_hash)")
    conn.commit()


def fetch_raw_documents(conn: sqlite3.Connection, min_words: int) -> pd.DataFrame:
    """Lê documentos brutos e filtra por contagem mínima de palavras."""
    df = pd.read_sql_query(
        "SELECT id AS original_id, url, content, content_type FROM documents",
        conn
    )
    df["content"] = df["content"].fillna("")
    wc = df["content"].apply(lambda s: len(re.findall(r"\w+", s)))
    df = df[wc >= min_words].copy()
    return df


# --------------------------------------------------------------------------------------
# Limpeza / NLP
# --------------------------------------------------------------------------------------
def basic_clean(text: str) -> str:
    """Lowercase, remove acentos/pontuação e colapsa espaços."""
    if not text:
        return ""
    t = unidecode(text.lower())
    t = _RX_PUNCT.sub(" ", t)
    t = _RX_SPACES.sub(" ", t)
    return t.strip()


@lru_cache(maxsize=4)
def get_stopwords(lang_prefix: str) -> set:
    """Baixa e cacheia stopwords por idioma."""
    import nltk
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords", quiet=True)
    from nltk.corpus import stopwords
    if lang_prefix.startswith("pt"):
        return set(stopwords.words("portuguese"))
    return set(stopwords.words("english"))


def tokenize_remove_stop(text: str, lang: str) -> str:
    sw = get_stopwords(lang[:2])
    tokens = [w for w in text.split() if w not in sw and not w.isdigit()]
    return " ".join(tokens)


def detect_lang_safe(text: str) -> str:
    """Tenta detectar idioma; fallback simples PT/EN."""
    if not text or len(text) < 20:
        return "pt"
    try:
        return detect(text)
    except Exception:
        return "pt" if re.search(r"[ãõáéíóúç]", text, re.IGNORECASE) else "en"


def lemmatize_optional(text: str, lang: str) -> str:
    """Lematiza se spaCy do idioma estiver instalado; caso contrário, retorna o texto."""
    try:
        import spacy
        model = "pt_core_news_sm" if lang.startswith("pt") else "en_core_web_sm"
        nlp = spacy.load(model)
        doc = nlp(text)
        return " ".join(t.lemma_ for t in doc if not t.is_punct and not t.is_space)
    except Exception:
        return text


def categorize(text_clean: str) -> str:
    """Marca categoria por regra (keyword matching)."""
    scores = {k: 0 for k in CATEGORIES}
    for cat, kws in CATEGORIES.items():
        for kw in kws:
            if kw in text_clean:
                scores[cat] += 1
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "geral"


# --------------------------------------------------------------------------------------
# Deduplicação
# --------------------------------------------------------------------------------------
def hash_text(text: str) -> str:
    import hashlib
    return hashlib.blake2b(text.encode("utf-8"), digest_size=16).hexdigest()


def deduplicate(df_clean: pd.DataFrame, sim_threshold: float = 0.92) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Remove duplicatas exatas (hash) e quase-duplicatas por similaridade TF‑IDF (char n-grams).
    Retorna (df_kept, dup_map).
    """
    # 1) exatas
    df = df_clean.copy()
    df["text_hash"] = df["content_clean"].apply(hash_text)
    before = len(df)
    df_kept = df.drop_duplicates(subset=["text_hash"]).copy()
    exact_removed = before - len(df_kept)

    # 2) quase-duplicatas
    to_drop: set[int] = set()
    if len(df_kept) >= 3:
        vec = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=2)
        X = vec.fit_transform(df_kept["content_clean"])
        if X.shape[0] >= 2 and X.nnz > 0:
            sim = cosine_similarity(X, dense_output=False)
            kept_idx = df_kept.index.to_list()
            lengths = df_kept["content_clean"].str.len()
            for i in range(sim.shape[0]):
                if kept_idx[i] in to_drop:
                    continue
                row = sim.getrow(i)
                # candidatos j>i acima do threshold
                mask = (row.data > sim_threshold) & (row.indices > i)
                for j in row.indices[mask]:
                    ii, jj = kept_idx[i], kept_idx[j]
                    keep = ii if lengths.loc[ii] >= lengths.loc[jj] else jj
                    drop = jj if keep == ii else ii
                    to_drop.add(drop)

    df_final = df_kept.drop(index=list(to_drop)).copy()

    # Mapa de duplicatas -> mantidos
    dup_rows: List[Dict[str, int]] = []
    # exatas
    for _, group in df_kept.groupby("text_hash"):
        kept_id = int(group.iloc[0]["original_id"])
        for _, r in group.iloc[1:].iterrows():
            dup_rows.append({"original_id": int(r["original_id"]), "kept_original_id": kept_id})
    # quase-dup (associa ao mais similar remanescente)
    if to_drop:
        vec2 = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=2).fit(df_final["content_clean"])
        X_final = vec2.transform(df_final["content_clean"])
        for drop_idx in to_drop:
            row = df_kept.loc[drop_idx]
            v = vec2.transform([row["content_clean"]])
            sims = cosine_similarity(v, X_final).ravel()
            kept_original = int(df_final.iloc[int(np.argmax(sims))]["original_id"])
            dup_rows.append({"original_id": int(row["original_id"]), "kept_original_id": kept_original})

    logging.info("[dedup] exatas removidas: %d | quase-dup removidas: %d", exact_removed, len(to_drop))
    dup_map = pd.DataFrame(dup_rows, columns=["original_id", "kept_original_id"])
    return df_final, dup_map


# --------------------------------------------------------------------------------------
# Embeddings
# --------------------------------------------------------------------------------------
def build_tfidf(df: pd.DataFrame) -> None:
    vectorizer = TfidfVectorizer(max_features=100_000, ngram_range=(1, 2), min_df=2)
    X = vectorizer.fit_transform(df["content_clean"])
    joblib.dump(vectorizer, ARTIFACTS_DIR / "tfidf_vectorizer.pkl")
    from scipy import sparse
    sparse.save_npz(ARTIFACTS_DIR / "tfidf_matrix.npz", X)
    logging.info("[emb] TF-IDF: %s salvo em %s", X.shape, ARTIFACTS_DIR)


def build_sbert(df: pd.DataFrame, model_name: str, batch_size: int = 32) -> None:
    model = load_sbert(model_name)
    embs = model.encode(
        df["content_clean"].tolist(),
        convert_to_numpy=True,
        show_progress_bar=True,
        batch_size=batch_size
    )
    np.save(ARTIFACTS_DIR / "sbert_embeddings.npy", embs)
    (ARTIFACTS_DIR / "sbert_model.txt").write_text(model_name, encoding="utf-8")
    logging.info("[emb] SBERT: %s salvo em %s", embs.shape, ARTIFACTS_DIR)


# --------------------------------------------------------------------------------------
# Persistência
# --------------------------------------------------------------------------------------
def upsert_processed(conn: sqlite3.Connection, df: pd.DataFrame, dup_map: pd.DataFrame) -> None:
    init_processed_schema(conn)
    cur = conn.cursor()

    # 1) Inserção em lote dos documentos processados
    now_iso = datetime.now(timezone.utc).isoformat()  # opcional: timestamp estável por execução
    rows = (
        (
            int(r.original_id),
            r.url,
            r.lang,
            r.category,
            r.content_clean,
            int(len(r.content_clean)),
            # se já existir a coluna text_hash no df (setada em deduplicate), use-a; caso não, gere aqui:
            getattr(r, "text_hash", hash_text(r.content_clean))
            ,
            now_iso,
        )
        for r in df.itertuples(index=False)
    )

    cur.executemany(
        """
        INSERT OR REPLACE INTO processed_documents
        (original_id, url, lang, category, content_clean, content_len, text_hash, processed_at)
        VALUES (?,?,?,?,?,?,?,?)
        """,
        rows,
    )

    # 2) Mapeamento de duplicatas (se houver)
    if not dup_map.empty:
        dup_rows = ((int(t.original_id), int(t.kept_original_id)) for t in dup_map.itertuples(index=False))
        cur.executemany(
            "INSERT INTO duplicates_map (original_id, kept_original_id) VALUES (?,?)",
            dup_rows,
        )

    conn.commit()

# --------------------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class PipelineConfig:
    min_words: int = 80
    sim_threshold: float = 0.92
    use_sbert: bool = True
    sbert_model: str = "paraphrase-multilingual-MiniLM-L12-v2"
    sbert_batch_size: int = 32


def run_pipeline(cfg: PipelineConfig) -> None:
    """Executa a etapa 02 completa: limpeza -> dedup -> persistência -> embeddings."""
    ensure_raw_db_exists()

    with closing(sqlite3.connect(str(RAW_DB_PATH))) as raw_conn, \
         closing(sqlite3.connect(str(PROCESSED_DB_PATH))) as proc_conn:

        init_processed_schema(proc_conn)

        raw = fetch_raw_documents(raw_conn, min_words=cfg.min_words)
        if raw.empty:
            logging.warning("Nenhum documento bruto com tamanho >= %d palavras.", cfg.min_words)
            return

        # Limpeza, idioma, stopwords, lematização, categorização
        rows: List[Dict[str, str | int]] = []
        for _, r in tqdm(raw.iterrows(), total=len(raw), desc="limpeza", unit="doc"):
            lang = detect_lang_safe(r["content"])
            t0 = basic_clean(r["content"])
            t1 = tokenize_remove_stop(t0, lang)
            t2 = lemmatize_optional(t1, lang)
            cat = categorize(t2)
            rows.append({
                "original_id": int(r["original_id"]),
                "url": r["url"],
                "lang": lang,
                "content_clean": t2,
                "category": cat,
            })
        df_clean = pd.DataFrame(rows)

        # Deduplicação
        df_kept, dup_map = deduplicate(df_clean, sim_threshold=cfg.sim_threshold)

        # Persistência
        upsert_processed(proc_conn, df_kept, dup_map)

        # Embeddings
        build_tfidf(df_kept)
        if cfg.use_sbert:
            build_sbert(df_kept, model_name=cfg.sbert_model, batch_size=cfg.sbert_batch_size)

        # Resumo
        total_docs = raw_conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        total_proc = proc_conn.execute("SELECT COUNT(*) FROM processed_documents").fetchone()[0]
        logging.info(
            "[ok] brutos=%d | processados=%d | db_processado=%s | artifacts=%s",
            total_docs, total_proc, PROCESSED_DB_PATH, ARTIFACTS_DIR
        )


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
def parse_args() -> PipelineConfig:
    ap = argparse.ArgumentParser(description="Etapa 02: Processamento e Tratamento de Dados")
    ap.add_argument("--min-words", type=int, default=80, help="Mínimo de palavras no bruto para processar.")
    ap.add_argument("--sim-threshold", type=float, default=0.92, help="Limiar de similaridade para quase-duplicatas.")
    ap.add_argument("--use-sbert", type=lambda v: str(v).lower() in {"1", "true", "t", "yes", "y"}, default=True,
                    help="Ativa embeddings SBERT (padrão: True).")
    ap.add_argument("--sbert-model", type=str, default="paraphrase-multilingual-MiniLM-L12-v2",
                    help="Nome do modelo SBERT.")
    ap.add_argument("--sbert-batch-size", type=int, default=32, help="Batch size para encode SBERT.")
    ap.add_argument("--log-level", type=str, default="INFO", help="Nível de log: DEBUG/INFO/WARNING/ERROR.")

    args = ap.parse_args()
    level = getattr(logging, args.log_level.upper(), logging.INFO)
    setup_logging(level)

    return PipelineConfig(
        min_words=args.min_words,
        sim_threshold=args.sim_threshold,
        use_sbert=args.use_sbert,
        sbert_model=args.sbert_model,
        sbert_batch_size=args.sbert_batch_size,
    )


if __name__ == "__main__":
    cfg = parse_args()
    run_pipeline(cfg)
