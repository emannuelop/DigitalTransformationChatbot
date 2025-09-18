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

# ---------------------------------------------------------------------
# Caminhos e setup
# ---------------------------------------------------------------------
BASE_DIR: Path = Path(__file__).resolve().parent.parent          # chatbot/
RAW_DB: Path = BASE_DIR / "crawler" / "data" / "knowledge_base.db"

DATA_DIR: Path = BASE_DIR / "data"                               # saída final
DATA_DIR.mkdir(exist_ok=True)
PROC_DB: Path = DATA_DIR / "knowledge_base_processed.db"
ART_DIR: Path = DATA_DIR / "artifacts"
ART_DIR.mkdir(parents=True, exist_ok=True)

DetectorFactory.seed = 42  # reprodutibilidade para langdetect

# Categorias simples por keywords (ajustável)
CATEGORIES: Dict[str, List[str]] = {
    "estrategia": ["estrategia", "governanca", "maturidade", "planejamento", "roadmap"],
    "dados_ia": ["dados", "ia", "inteligencia artificial", "machine learning", "analytics", "algoritmo"],
    "processos": ["processo", "bpmn", "workflow", "automacao", "otimizacao"],
    "infra_seg": ["cloud", "nuvem", "seguranca", "lgpd", "api", "arquitetura"],
    "servicos_digitais": ["servico digital", "gov.br", "portal", "ux", "acessibilidade"],
    "setor_publico": ["governo digital", "politica publica", "decreto", "licitacao"],
}

# Regex pré-compiladas para limpeza
_RX_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
_RX_SPACES = re.compile(r"\s+")


# ---------------------------------------------------------------------
# Utils
# ---------------------------------------------------------------------
def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO),
                        format="[%(levelname)s] %(message)s")


def ensure_raw_db() -> None:
    if not RAW_DB.exists():
        raise FileNotFoundError(f"DB bruto não encontrado: {RAW_DB}. Rode o crawler antes.")


def basic_clean(text: str) -> str:
    """Lowercase, remove acento/pontuação, normaliza espaços."""
    if not text:
        return ""
    t = unidecode(text.lower())
    t = _RX_PUNCT.sub(" ", t)
    t = _RX_SPACES.sub(" ", t)
    return t.strip()


def detect_lang_safe(text: str) -> str:
    """PT/EN com fallback simples."""
    if not text or len(text) < 20:
        return "pt"
    try:
        return detect(text)
    except Exception:
        return "pt" if re.search(r"[ãõáéíóúç]", text, re.IGNORECASE) else "en"


@lru_cache(maxsize=4)
def get_stopwords(prefix: str) -> set:
    """Carrega stopwords NLTK e cacheia por idioma."""
    import nltk
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords", quiet=True)
    from nltk.corpus import stopwords
    return set(stopwords.words("portuguese" if prefix.startswith("pt") else "english"))


def remove_stopwords(text: str, lang: str) -> str:
    sw = get_stopwords(lang[:2])
    return " ".join(w for w in text.split() if w not in sw and not w.isdigit())


def lemmatize_optional(text: str, lang: str) -> str:
    """Lematiza se spaCy estiver instalado; caso contrário, retorna o próprio texto."""
    try:
        import spacy
        nlp = spacy.load("pt_core_news_sm" if lang.startswith("pt") else "en_core_web_sm")
        return " ".join(t.lemma_ for t in nlp(text) if not t.is_punct and not t.is_space)
    except Exception:
        return text


def categorize(text_clean: str) -> str:
    scores = {k: 0 for k in CATEGORIES}
    for cat, kws in CATEGORIES.items():
        for kw in kws:
            if kw in text_clean:
                scores[cat] += 1
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "geral"


def hash_text(text: str) -> str:
    import hashlib
    return hashlib.blake2b(text.encode("utf-8"), digest_size=16).hexdigest()


# ---------------------------------------------------------------------
# Leitura do bruto (PDF-only)
# ---------------------------------------------------------------------
def fetch_raw_pdfs(conn: sqlite3.Connection, min_words: int) -> pd.DataFrame:
    """Carrega documentos do banco bruto e aplica filtro por tamanho."""
    cols = pd.read_sql_query("PRAGMA table_info(documents);", conn)["name"].tolist()
    select_cols = ["id AS original_id", "url", "content"]
    if "content_type" in cols:  # compatível com versões antigas do crawler
        select_cols.append("content_type")

    df = pd.read_sql_query(f"SELECT {', '.join(select_cols)} FROM documents", conn)
    df["content"] = df["content"].fillna("")
    wc = df["content"].apply(lambda s: len(re.findall(r"\w+", s)))
    return df[(wc >= min_words) & (df["content"].str.len() > 0)].copy()


# ---------------------------------------------------------------------
# Deduplicação
# ---------------------------------------------------------------------
def deduplicate(df_clean: pd.DataFrame, sim_threshold: float = 0.92) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Remove duplicatas exatas (hash) e quase-duplicatas (TF-IDF char n-grams).
    Retorna (df_kept, dup_map).
    """
    df = df_clean.copy()
    df["text_hash"] = df["content_clean"].apply(hash_text)

    # Exatas
    before = len(df)
    df_kept = df.drop_duplicates(subset=["text_hash"]).copy()
    exact_removed = before - len(df_kept)

    # Quase-duplicatas
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
                mask = (row.data > sim_threshold) & (row.indices > i)
                for j in row.indices[mask]:
                    ii, jj = kept_idx[i], kept_idx[j]
                    keep = ii if lengths.loc[ii] >= lengths.loc[jj] else jj
                    drop = jj if keep == ii else ii
                    to_drop.add(drop)

    df_final = df_kept.drop(index=list(to_drop)).copy()

    # Mapa de duplicatas → mantidos
    dup_rows: List[Dict[str, int]] = []
    for _, group in df.groupby("text_hash"):
        kept_original = int(group.iloc[0]["original_id"])
        for _, r in group.iloc[1:].iterrows():
            dup_rows.append({"original_id": int(r["original_id"]), "kept_original_id": kept_original})

    if to_drop:
        vec2 = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=2).fit(df_final["content_clean"])
        Xf = vec2.transform(df_final["content_clean"])
        for idx in to_drop:
            row = df_kept.loc[idx]
            v = vec2.transform([row["content_clean"]])
            sims = cosine_similarity(v, Xf).ravel()
            kept_original = int(df_final.iloc[int(np.argmax(sims))]["original_id"])
            dup_rows.append({"original_id": int(row["original_id"]), "kept_original_id": kept_original})

    logging.info("[dedup] exatas removidas: %d | quase-dup removidas: %d", exact_removed, len(to_drop))
    dup_map = pd.DataFrame(dup_rows, columns=["original_id", "kept_original_id"])
    return df_final, dup_map


# ---------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------
def build_tfidf(df: pd.DataFrame) -> None:
    vectorizer = TfidfVectorizer(max_features=100_000, ngram_range=(1, 2), min_df=2)
    X = vectorizer.fit_transform(df["content_clean"])
    joblib.dump(vectorizer, ART_DIR / "tfidf_vectorizer.pkl")
    from scipy import sparse
    sparse.save_npz(ART_DIR / "tfidf_matrix.npz", X)
    logging.info("[emb] TF-IDF: %s salvo em %s", X.shape, ART_DIR)


def build_sbert(df: pd.DataFrame, model_name: str, batch: int = 32) -> None:
    from sentence_transformers import SentenceTransformer  # lazy import
    model = SentenceTransformer(model_name)
    embs = model.encode(df["content_clean"].tolist(),
                        convert_to_numpy=True, show_progress_bar=True, batch_size=batch)
    np.save(ART_DIR / "sbert_embeddings.npy", embs)
    (ART_DIR / "sbert_model.txt").write_text(model_name, encoding="utf-8")
    logging.info("[emb] SBERT: %s salvo em %s", embs.shape, ART_DIR)


# ---------------------------------------------------------------------
# Persistência
# ---------------------------------------------------------------------
def init_schema(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS processed_documents(
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
        CREATE TABLE IF NOT EXISTS duplicates_map(
            original_id INTEGER,
            kept_original_id INTEGER
        )
    """)
    conn.commit()


def upsert_processed(conn: sqlite3.Connection, df: pd.DataFrame, dup_map: pd.DataFrame) -> None:
    init_schema(conn)
    cur = conn.cursor()
    now_iso = datetime.now(timezone.utc).isoformat()

    rows = (
        (
            int(r.original_id), r.url, r.lang, r.category, r.content_clean,
            int(len(r.content_clean)), getattr(r, "text_hash", hash_text(r.content_clean)), now_iso
        )
        for r in df.itertuples(index=False)
    )

    cur.executemany("""
        INSERT OR REPLACE INTO processed_documents
        (original_id, url, lang, category, content_clean, content_len, text_hash, processed_at)
        VALUES (?,?,?,?,?,?,?,?)
    """, rows)

    if not dup_map.empty:
        cur.executemany(
            "INSERT INTO duplicates_map (original_id, kept_original_id) VALUES (?,?)",
            ((int(t.original_id), int(t.kept_original_id)) for t in dup_map.itertuples(index=False))
        )

    conn.commit()


# ---------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class Config:
    min_words: int = 80
    sim_threshold: float = 0.92
    use_sbert: bool = True
    sbert_model: str = "paraphrase-multilingual-MiniLM-L12-v2"
    sbert_batch: int = 32
    log_level: str = "INFO"


def run(cfg: Config) -> None:
    """Executa: carregar -> limpar -> deduplicar -> persistir -> embeddings."""
    ensure_raw_db()
    logging.getLogger().setLevel(getattr(logging, cfg.log_level.upper(), logging.INFO))

    with closing(sqlite3.connect(str(RAW_DB))) as raw, \
         closing(sqlite3.connect(str(PROC_DB))) as proc:

        init_schema(proc)

        # Seleção
        raw_df = fetch_raw_pdfs(raw, min_words=cfg.min_words)
        if raw_df.empty:
            logging.warning("Nenhum PDF com pelo menos %d palavras", cfg.min_words)
            return

        # Limpeza / transformação
        rows: List[Dict[str, str | int]] = []
        for _, rec in tqdm(raw_df.iterrows(), total=len(raw_df), desc="limpeza", unit="pdf"):
            lang = detect_lang_safe(rec["content"])
            t0 = basic_clean(rec["content"])
            t1 = remove_stopwords(t0, lang)
            t2 = lemmatize_optional(t1, lang)  # opcional (se spaCy não estiver, mantém t1)
            rows.append({
                "original_id": int(rec["original_id"]),
                "url": rec["url"],
                "lang": lang,
                "content_clean": t2,
                "category": categorize(t2),
            })
        clean_df = pd.DataFrame(rows)

        # Deduplicação + persistência
        kept_df, dup_map = deduplicate(clean_df, sim_threshold=cfg.sim_threshold)
        upsert_processed(proc, kept_df, dup_map)

        # Embeddings
        build_tfidf(kept_df)
        if cfg.use_sbert:
            build_sbert(kept_df, cfg.sbert_model, cfg.sbert_batch)

        # Resumo
        n_raw = raw.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        n_proc = proc.execute("SELECT COUNT(*) FROM processed_documents").fetchone()[0]
        logging.info("[ok] brutos=%d | processados=%d | db=%s | artifacts=%s",
                     n_raw, n_proc, PROC_DB, ART_DIR)


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def parse_args() -> Config:
    ap = argparse.ArgumentParser(description="Etapa 02 (KDD) - Processamento PDF-only")
    ap.add_argument("--min-words", type=int, default=80)
    ap.add_argument("--sim-threshold", type=float, default=0.92)
    ap.add_argument("--use-sbert", type=lambda v: str(v).lower() in {"1", "true", "t", "yes", "y"}, default=True)
    ap.add_argument("--sbert-model", type=str, default="paraphrase-multilingual-MiniLM-L12-v2")
    ap.add_argument("--sbert-batch", type=int, default=32)
    ap.add_argument("--log-level", type=str, default="INFO")
    args = ap.parse_args()
    setup_logging(args.log_level)
    return Config(args.min_words, args.sim_threshold, args.use_sbert,
                  args.sbert_model, args.sbert_batch, args.log_level)


if __name__ == "__main__":
    run(parse_args())
