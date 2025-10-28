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

# Chunks
CHUNK_SIZE_CHARS: int    = 1000
CHUNK_OVERLAP_CHARS: int = 150
MAX_CHARS_PER_CHUNK: int = 900

# Recuperação (mais rígido para evitar “falsos positivos”)
TOP_K: int          = 6
SCORE_CUTOFF: float = 0.30  # ↑ mais estrito

# Reranking híbrido (vetorial + léxico)
HYBRID_LAMBDA: float = 0.25  # 0 = só vetorial; 1 = só léxico

# LM Studio
LMSTUDIO_HOST: str  = "http://127.0.0.1:1234"
LMSTUDIO_MODEL: str = "ibm/granite-4-h-tiny"

# Geração (mais determinística para respostas consistentes)
GEN_TEMPERATURE: float = 0.05
GEN_TOP_P: float       = 0.75
GEN_MAX_TOKENS: int    = 1024
GEN_TIMEOUT_S: int     = 240
GEN_RETRIES: int       = 2
GEN_BACKOFF_S: float   = 1.5
GEN_STOP: list[str] | None = None

# Domain guard (tema padrão do produto)
DOMAIN_GUARD_STRICT: bool = True            # mantém foco em setor público por padrão
# Bypass quando o contexto for de PDF enviado pelo usuário
USER_UPLOAD_BYPASS_GUARD: bool = True       # permite “tema livre” quando a fonte é user_sources
MIN_ANSWER_COVERAGE_USER: float = 0.05      # cobertura mínima quando for PDF do usuário (mais suave)

# Força PT-BR sempre
FORCE_PT_BR: bool = False

# Controle de resposta
ANSWER_SENTINEL: str      = "<FIM/>"
CONTINUE_MAX_ROUNDS: int  = 3
MIN_GENERATION_CHARS: int = 500            # padrão (domínio transformação digital)
MIN_GENERATION_CHARS_USER: int = 180       # mínimo mais curto para PDF do usuário

# Judge (desligado)
JUDGE_ANSWERABILITY: bool = False
JUDGE_MAX_TOKENS: int     = 4

# Relevância léxica (gates determinísticos)
KEYWORD_MIN_LEN: int        = 5
KEYWORD_MIN_HITS: int       = 2
MIN_ANSWER_COVERAGE: float  = 0.10

def _validate() -> None:
    for p in (DATA_DIR, ART_DIR):
        p.mkdir(parents=True, exist_ok=True)
_validate()
