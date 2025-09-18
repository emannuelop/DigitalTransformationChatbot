import sqlite3
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from sentence_transformers import SentenceTransformer
from .settings import PROC_DB, ART_DIR, EMB_NPY, MAPPING_PARQUET

# ------------------------------------------------------------
# Funções auxiliares
# ------------------------------------------------------------
def chunk_text(t: str, size: int = 1600, overlap: int = 200):
    """Divide um texto longo em janelas menores com overlap."""
    t = str(t or "")
    if len(t) <= size:
        return [t]
    chunks = []
    i = 0
    while i < len(t):
        chunks.append(t[i:i+size])
        i += size - overlap
    return chunks

def _table_columns(cur, table):
    cur.execute(f"PRAGMA table_info({table});")
    return [r[1] for r in cur.fetchall()]

def _pick_first_present(columns, candidates):
    for c in candidates:
        if c in columns:
            return c
    return None

def detect_table_and_columns(conn):
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [r[0] for r in cur.fetchall()]
    if not tables:
        raise RuntimeError("Nenhuma tabela encontrada no DB processado.")

    table = "processed_documents" if "processed_documents" in tables else tables[0]
    cols = _table_columns(cur, table)

    id_col    = "id" if "id" in cols else None
    # tenta vários nomes comuns; se não achar, escolhe heurística depois
    text_col  = _pick_first_present(cols, ["text", "clean_text", "content", "body", "texto", "conteudo"])
    title_col = _pick_first_present(cols, ["title", "page_title", "document_title", "name", "titulo"])
    lang_col  = _pick_first_present(cols, ["lang", "language", "idioma"])
    url_col   = _pick_first_present(cols, ["url", "source_url", "link", "href"])

    # se não achou text_col por nome, tenta heurística em uma amostra
    if not text_col:
        col_list = ", ".join([f'"{c}"' for c in cols])
        sample = pd.read_sql_query(f'SELECT {col_list} FROM "{table}" ORDER BY rowid LIMIT 200;', conn)
        str_cols = [c for c in sample.columns if pd.api.types.is_string_dtype(sample[c])]
        if str_cols:
            # escolhe a coluna string com maior média de caracteres
            lens = [(sample[c].fillna("").astype(str).str.len().mean(), c) for c in str_cols]
            lens.sort(reverse=True)
            text_col = lens[0][1]

    if not text_col:
        raise RuntimeError(f"Não foi possível identificar a coluna de texto na tabela '{table}'.")

    return table, {"id": id_col, "text": text_col, "title": title_col,
                   "lang": lang_col, "url": url_col}

def load_df():
    conn = sqlite3.connect(PROC_DB)
    table, cols = detect_table_and_columns(conn)

    sel = []
    for key in ["id", "title", "text", "lang", "url"]:
        col = cols.get(key)
        if col:
            sel.append(f'"{col}" AS {key}')
        else:
            sel.append(f"'' AS {key}")
    query = f'SELECT {", ".join(sel)} FROM "{table}" '
    # ordene por id se existir; senão, rowid
    if cols.get("id"):
        query += "ORDER BY id"
    else:
        query += "ORDER BY rowid"

    df = pd.read_sql_query(query, conn)
    conn.close()

    for c in ["title", "text", "lang", "url"]:
        df[c] = df[c].fillna("").astype(str)
    if df["id"].isnull().all() or (df["id"] == "").all():
        df["id"] = np.arange(1, len(df)+1)

    return df

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    base_df = load_df()
    print(f"[embedder] Documentos originais: {len(base_df)}")
    print("[embedder] amostra de comprimentos de texto:", base_df["text"].str.len().describe().to_dict())

    # chunking
    rows = []
    for _, r in base_df.iterrows():
        parts = chunk_text(r["text"], size=1600, overlap=200)
        for j, p in enumerate(parts):
            rows.append({
                "id": f'{r["id"]}_{j+1}',
                "title": r["title"],
                "text": p,
                "lang": r["lang"],
                "url": r["url"],
            })
    df = pd.DataFrame(rows)
    print(f"[embedder] Após chunking: {len(df)} segmentos")
    print("[embedder] comprimentos pós-chunk:", df["text"].str.len().describe().to_dict())

    # embeddings
    model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    model = SentenceTransformer(model_name)

    texts = df["text"].tolist()
    batch = 256
    embs = []
    for i in tqdm(range(0, len(texts), batch), desc="Embeddings"):
        chunk = texts[i:i+batch]
        v = model.encode(chunk, normalize_embeddings=True, show_progress_bar=False)
        embs.append(v.astype(np.float32))
    embs = np.vstack(embs)

    ART_DIR.mkdir(parents=True, exist_ok=True)
    np.save(EMB_NPY, embs)
    df.to_parquet(MAPPING_PARQUET, index=False)

    print(f"[OK] Embeddings: {EMB_NPY} | shape={embs.shape}")
    print(f"[OK] Mapping:    {MAPPING_PARQUET}")

if __name__ == "__main__":
    main()
