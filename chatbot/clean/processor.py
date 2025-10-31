import logging
import re
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from langdetect import DetectorFactory, detect
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from unidecode import unidecode

BASE_DIR: Path = Path(__file__).resolve().parent.parent         
RAW_DB:   Path = BASE_DIR / "extraction" / "data" / "knowledge_base.db"

DATA_DIR: Path = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

PROC_DB: Path = DATA_DIR / "knowledge_base_processed.db"

ART_DIR:  Path = DATA_DIR / "artifacts"
ART_DIR.mkdir(parents=True, exist_ok=True)

MIN_WORDS       = 80
SIM_THRESHOLD   = 0.92
USE_SBERT       = True
SBERT_MODEL     = "paraphrase-multilingual-MiniLM-L12-v2"
SBERT_BATCH     = 32
LOG_LEVEL       = "INFO"

DetectorFactory.seed = 42  

CATEGORIES: Dict[str, List[str]] = {
    "estrategia":       ["estrategia", "governanca", "maturidade", "planejamento", "roadmap"],
    "dados_ia":         ["dados", "ia", "inteligencia artificial", "machine learning", "analytics", "algoritmo"],
    "processos":        ["processo", "bpmn", "workflow", "automacao", "otimizacao"],
    "infra_seg":        ["cloud", "nuvem", "seguranca", "lgpd", "api", "arquitetura"],
    "servicos_digitais":["servico digital", "gov.br", "portal", "ux", "acessibilidade"],
    "setor_publico":    ["governo digital", "politica publica", "decreto", "licitacao"],
}

PT_STOP = {
    "a","à","às","ao","aos","aquele","aquela","aquilo","as","até","com","como","da","das","de","dela","dele",
    "deles","demais","depois","do","dos","e","ela","elas","ele","eles","em","entre","era","eram","esse","essa",
    "isso","esta","este","isto","foi","foram","há","já","la","lhe","lhes","mais","mas","me","mesmo",
    "meu","minha","muito","na","nas","não","nem","no","nos","nós","o","os","ou","para","pela","pelas","pelo",
    "pelos","por","porque","qual","quando","que","quem","se","sem","seu","sua","suas","são","tão","também",
    "te","tem","têm","um","uma","você","vocês"
}

_RX_PUNCT  = re.compile(r"[^\w\s]", re.UNICODE)
_RX_SPACES = re.compile(r"\s+")
_RX_ACCENT_PT = re.compile(r"[ãõáéíóúâêîôûç]", re.IGNORECASE)

def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO),
                        format="[%(levelname)s] %(message)s")

def ensure_raw_db() -> None:
    if not RAW_DB.exists():
        raise FileNotFoundError(f"DB bruto não encontrado: {RAW_DB}. Rode o scraping antes.")

def basic_clean(text: str) -> str:
    if not text:
        return ""
    t = unidecode(text.lower())
    t = _RX_PUNCT.sub(" ", t)
    t = _RX_SPACES.sub(" ", t)
    return t.strip()

def detect_lang_safe(text: str) -> str:
    if not text or len(text) < 20:
        return "pt"
    try:
        return detect(text)
    except Exception:
        return "pt" if _RX_ACCENT_PT.search(text or "") else "en"

def is_pt(text: str) -> bool:
    lang = (detect_lang_safe(text) or "").lower()
    if lang.startswith("pt"):
        return True
    t = (text or "").lower()
    has_accents = bool(_RX_ACCENT_PT.search(t))
    sw_hits = sum(1 for w in PT_STOP if f" {w} " in f" {t} ")
    return has_accents or sw_hits >= 3

def remove_stopwords_pt(text: str) -> str:
    return " ".join(w for w in text.split() if w not in PT_STOP and not w.isdigit())

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

def fetch_raw_pdfs(conn: sqlite3.Connection, min_words: int) -> pd.DataFrame:
    """
    Busca documentos do banco RAW (knowledge_base.db).
    Compatível com a nova coluna user_id em user_sources (não afeta esta query).
    """
    cols = pd.read_sql_query("PRAGMA table_info(documents);", conn)["name"].tolist()
    select_cols = ["id AS original_id", "url", "content"]
    if "content_type" in cols:  
        select_cols.append("content_type")
    if "user_id" in cols:
        select_cols.append("user_id")
    else:
        select_cols.append("NULL AS user_id") # Para compatibilidade

    df = pd.read_sql_query(f"SELECT {', '.join(select_cols)} FROM documents", conn)
    df["content"] = df["content"].fillna("")
    wc = df["content"].apply(lambda s: len(re.findall(r"\w+", s)))
    return df[(wc >= min_words) & (df["content"].str.len() > 0)].copy()

def deduplicate(df_clean: pd.DataFrame, sim_threshold: float = 0.92) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Remove duplicatas exatas e quase-duplicatas usando TF-IDF.
    NOTA: TF-IDF aqui é usado APENAS para deduplicação, não para o RAG.
    """
    df = df_clean.copy()
    df["text_hash"] = df["content_clean"].apply(hash_text)

    before = len(df)
    df_kept = df.drop_duplicates(subset=["text_hash"]).copy()
    exact_removed = before - len(df_kept)

    to_drop: set[int] = set()
    if len(df_kept) >= 3:
        # TF-IDF usado APENAS para detectar duplicatas (não salvo em disco)
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
                for j, s in zip(row.indices, row.data):
                    if j <= i or s <= sim_threshold:
                        continue
                    ii, jj = kept_idx[i], kept_idx[j]
                    keep = ii if lengths.loc[ii] >= lengths.loc[jj] else jj
                    drop = jj if keep == ii else ii
                    to_drop.add(drop)

    df_final = df_kept.drop(index=list(to_drop)).copy()

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

# REMOVIDO: build_tfidf() - não é usado no RAG
# O TF-IDF é usado apenas internamente na função deduplicate() acima

def build_sbert(df: pd.DataFrame, model_name: str, batch: int = 32) -> None:
    """
    Gera embeddings SBERT (usado no RAG).
    Compatível com documentos de qualquer usuário (user_id não afeta).
    """
    if not USE_SBERT:
        logging.info("[emb] SBERT desativado.")
        return
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as e:
        logging.warning("[emb] SBERT indisponível (%s). Pulei.", e)
        return
    
    logging.info("[emb] Gerando embeddings SBERT para %d documentos...", len(df))
    model = SentenceTransformer(model_name)
    embs = model.encode(df["content_clean"].tolist(),
                        convert_to_numpy=True, show_progress_bar=True, batch_size=batch)
    np.save(ART_DIR / "sbert_embeddings.npy", embs)
    (ART_DIR / "sbert_model.txt").write_text(model_name, encoding="utf-8")
    logging.info("[emb] SBERT: %s salvo em %s", embs.shape, ART_DIR)

def init_schema(conn: sqlite3.Connection) -> None:
    """
    Cria schema do banco processado.
    NÃO precisa de user_id aqui (fica apenas no banco RAW).
    """
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
            processed_at TEXT,
            user_id INTEGER -- NULL para documentos globais, ID para documentos de usuário
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS duplicates_map(
            original_id INTEGER,
            kept_original_id INTEGER
        )
    """)
    # Adicionar user_id à processed_documents se não existir (para migração)
    cols = pd.read_sql_query("PRAGMA table_info(processed_documents);", conn)["name"].tolist()
    if "user_id" not in cols:
        cur.execute("ALTER TABLE processed_documents ADD COLUMN user_id INTEGER;")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_processed_documents_user_id ON processed_documents(user_id);")
    conn.commit()

def upsert_processed(conn: sqlite3.Connection, df: pd.DataFrame, dup_map: pd.DataFrame) -> None:
    """
    Salva documentos processados no banco.
    Compatível com documentos de qualquer usuário.
    """
    init_schema(conn)
    cur = conn.cursor()
    now_iso = datetime.now(timezone.utc).isoformat()

    rows = (
        (
            int(r.original_id), r.url, r.lang, r.category, r.content_clean,
            int(len(r.content_clean)), getattr(r, "text_hash", hash_text(r.content_clean)), now_iso, r.user_id
        )
        for r in df.itertuples(index=False)
    )

    cur.executemany("""
        INSERT OR REPLACE INTO processed_documents
        (original_id, url, lang, category, content_clean, content_len, text_hash, processed_at, user_id)
        VALUES (?,?,?,?,?,?,?,?,?)
    """, rows)

    if not dup_map.empty:
        cur.executemany(
            "INSERT INTO duplicates_map (original_id, kept_original_id) VALUES (?,?)",
            ((int(t.original_id), int(t.kept_original_id)) for t in dup_map.itertuples(index=False))
        )

    conn.commit()

def run() -> None:
    """
    Processa TODOS os documentos do banco RAW (base global + PDFs de usuários).
    Compatível com a nova coluna user_id (não afeta o processamento).
    """
    setup_logging(LOG_LEVEL)
    ensure_raw_db()

    with closing(sqlite3.connect(str(RAW_DB))) as raw, \
         closing(sqlite3.connect(str(PROC_DB))) as proc:

        init_schema(proc)

        raw_df = fetch_raw_pdfs(raw, min_words=MIN_WORDS)
        if raw_df.empty:
            logging.warning("Nenhum PDF com pelo menos %d palavras", MIN_WORDS)
            return

        rows: List[Dict[str, str | int]] = []
        dropped_lang = 0
        for _, rec in tqdm(raw_df.iterrows(), total=len(raw_df), desc="limpeza", unit="pdf"):
            content = rec["content"] or ""
            if not is_pt(content):
                dropped_lang += 1
                continue

            t0 = basic_clean(content)
            t1 = remove_stopwords_pt(t0)
            rows.append({
                "original_id": int(rec["original_id"]),
                "url": rec["url"],
                "lang": "pt",
                "content_clean": t1,
                "category": categorize(t1),
                "user_id": rec["user_id"] # Adicionar user_id
            })

        logging.info("[lang] descartados por idioma != pt: %d", dropped_lang)
        clean_df = pd.DataFrame(rows)
        if clean_df.empty:
            logging.warning("Após filtro de idioma, não sobraram documentos em PT.")
            return

        kept_df, dup_map = deduplicate(clean_df, sim_threshold=SIM_THRESHOLD)
        upsert_processed(proc, kept_df, dup_map)

        # REMOVIDO: build_tfidf() - não é usado no RAG
        # Apenas SBERT é gerado (usado pelo embedder e RAG)
        build_sbert(kept_df, SBERT_MODEL, SBERT_BATCH)

        n_raw  = raw.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        n_proc = proc.execute("SELECT COUNT(*) FROM processed_documents").fetchone()[0]
        logging.info("[ok] brutos=%d | processados=%d | db=%s | artifacts=%s",
                     n_raw, n_proc, PROC_DB, ART_DIR)

if __name__ == "__main__":
    run()
