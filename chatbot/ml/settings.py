from __future__ import annotations
from pathlib import Path

BASE_DIR: Path = Path(__file__).resolve().parents[1]
DATA_DIR: Path = BASE_DIR / "data"
ART_DIR:  Path = DATA_DIR / "artifacts"
ART_DIR.mkdir(parents=True, exist_ok=True)

PROC_DB: Path          = DATA_DIR / "knowledge_base_processed.db"
MAPPING_PARQUET: Path  = ART_DIR  / "sbert_mapping.parquet"
EMB_NPY: Path          = ART_DIR  / "sbert_embeddings.npy"
FAISS_INDEX: Path      = ART_DIR  / "faiss.index"

# Embeddings
SBERT_MODEL: str       = "paraphrase-multilingual-MiniLM-L12-v2"
SBERT_BATCH: int       = 32
SBERT_NORMALIZE: bool  = True

# Chunks (menores = prompt mais rápido)
CHUNK_SIZE_CHARS: int    = 1000
CHUNK_OVERLAP_CHARS: int = 150
MAX_CHARS_PER_CHUNK: int = 900

# Recuperação
TOP_K: int          = 2
SCORE_CUTOFF: float = 0.00

# LM Studio
LMSTUDIO_HOST: str  = "http://127.0.0.1:1234"
# Mais rápido que o reasoning:
LMSTUDIO_MODEL: str = "microsoft/phi-4-mini-instruct"

# Geração (o pipeline ajusta max_tokens dinamicamente até 1000)
GEN_TEMPERATURE: float = 0.15
GEN_TOP_P: float       = 0.90
GEN_MAX_TOKENS: int    = 450
GEN_TIMEOUT_S: int     = 240   # tempo maior evita timeout em cold start
GEN_RETRIES: int       = 2
GEN_BACKOFF_S: float   = 1.5
GEN_STOP: list[str] | None = None

# Força PT-BR sempre
FORCE_PT_BR: bool = True

def _validate() -> None:
    for p in (DATA_DIR, ART_DIR):
        p.mkdir(parents=True, exist_ok=True)
_validate()
