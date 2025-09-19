# chatbot/ml/build_index.py
import numpy as np
import faiss
from pathlib import Path
import pandas as pd

from .settings import EMB_NPY, FAISS_INDEX, MAPPING_PARQUET

def main():
    assert Path(EMB_NPY).exists(), f"Embeddings não encontrados: {EMB_NPY}"
    assert Path(MAPPING_PARQUET).exists(), f"Mapping não encontrado: {MAPPING_PARQUET}"

    embs = np.load(EMB_NPY)                     # (N, D) float32
    df = pd.read_parquet(MAPPING_PARQUET)       # valida alinhamento
    assert len(df) == embs.shape[0], f"Linhas mapping ({len(df)}) != embeddings ({embs.shape[0]})"

    dim = embs.shape[1]
    faiss.normalize_L2(embs)                    # deixa unit-norm (para IP ~ cos)
    index = faiss.IndexFlatIP(dim)
    index.add(embs.astype(np.float32))

    FAISS_INDEX.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(FAISS_INDEX))
    print(f"[OK] FAISS salvo em: {FAISS_INDEX}")
    print(f"[OK] N={embs.shape[0]} | dim={dim}")

if __name__ == "__main__":
    main()
