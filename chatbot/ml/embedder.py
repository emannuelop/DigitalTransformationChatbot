from contextlib import closing
import sqlite3

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from .settings import (
    PROC_DB, ART_DIR, SBERT_MODEL, SBERT_BATCH, MAPPING_PARQUET,
    CHUNK_SIZE_CHARS, CHUNK_OVERLAP_CHARS
)

def load_processed() -> pd.DataFrame:
    with closing(sqlite3.connect(str(PROC_DB))) as conn:
        return pd.read_sql_query(
            """
            SELECT original_id, url, content_clean
            FROM processed_documents
            WHERE content_clean IS NOT NULL AND TRIM(content_clean) <> ''
            """,
            conn,
        )

def chunk_text(t: str, size: int, overlap: int) -> list[str]:
    t = (t or "").strip()
    if not t:
        return []
    if len(t) <= size:
        return [t]
    chunks, i = [], 0
    step = max(1, size - overlap)
    while i < len(t):
        chunks.append(t[i : i + size])
        i += step
    return chunks

def main() -> None:
    base = load_processed()
    rows = []
    for _, r in tqdm(base.iterrows(), total=len(base), desc="chunking", unit="doc"):
        for ck in chunk_text(r["content_clean"], CHUNK_SIZE_CHARS, CHUNK_OVERLAP_CHARS):
            rows.append({
                "original_id": int(r["original_id"]),
                "url": r["url"],
                "text": ck
            })

    if not rows:
        raise RuntimeError("Nenhum chunk gerado — confira o banco processado.")

    # Mapping: RangeIndex corresponde à ordem de embeddings/FAISS
    map_df = pd.DataFrame(rows, copy=False)

    # >>> Novo: compatível com filtro multiusuário do rag_pipeline.search()
    # Mantemos como float (NaN) para permitir .isna()
    if "user_id" not in map_df.columns:
        map_df["user_id"] = np.nan

    # Embeddings
    model = SentenceTransformer(SBERT_MODEL)
    embs = model.encode(
        map_df["text"].tolist(),
        convert_to_numpy=True,
        show_progress_bar=True,
        batch_size=SBERT_BATCH,
        normalize_embeddings=True,
    )
    np.save(ART_DIR / "sbert_embeddings.npy", embs)

    # Sanity-check de alinhamento
    assert len(map_df) == embs.shape[0], f"Linhas mapping ({len(map_df)}) != embeddings ({embs.shape[0]})"

    # Mapping parquet (com URLs e user_id)
    map_df.to_parquet(MAPPING_PARQUET, index=False)

if __name__ == "__main__":
    main()
