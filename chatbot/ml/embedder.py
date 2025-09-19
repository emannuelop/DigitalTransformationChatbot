# chatbot/ml/embedder.py
from contextlib import closing
import sqlite3
from pathlib import Path

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
            rows.append({"original_id": int(r["original_id"]), "url": r["url"], "text": ck})

    map_df = pd.DataFrame(rows)
    assert not map_df.empty, "Nenhum chunk gerado — confira o banco processado."

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

    # Mapping parquet (com URLs)
    map_df.to_parquet(MAPPING_PARQUET, index=False)

if __name__ == "__main__":
    main()
