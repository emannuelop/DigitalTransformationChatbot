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

SBERT_MODEL: str  = "paraphrase-multilingual-MiniLM-L12-v2"  
SBERT_BATCH: int  = 32
SBERT_NORMALIZE: bool = True  

CHUNK_SIZE_CHARS: int    = 1600   
CHUNK_OVERLAP_CHARS: int = 200    

TOP_K: int          = 5     
SCORE_CUTOFF: float = 0.0   
MAX_CHARS_PER_CHUNK: int = 1600  


LMSTUDIO_HOST: str  = "http://127.0.0.1:1234"
LMSTUDIO_MODEL: str = "microsoft/phi-4-mini-reasoning"

GEN_TEMPERATURE: float = 0.2
GEN_TOP_P: float       = 0.9
GEN_MAX_TOKENS: int    = 512
GEN_TIMEOUT_S: int     = 15
GEN_RETRIES: int       = 1
GEN_BACKOFF_S: float   = 0.5
GEN_STOP: list[str] | None = None  

FORCE_PT_BR: bool = True   

def _validate() -> None:
    for p in (DATA_DIR, ART_DIR):
        if not p.exists():
            p.mkdir(parents=True, exist_ok=True)

_validate()
