from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]  # .../chatbot
DATA_DIR = BASE_DIR / "data"
ART_DIR = DATA_DIR / "artifacts"

# DBs e artefatos
PROC_DB = DATA_DIR / "knowledge_base_processed.db"

# Saídas/artefatos desta etapa
EMB_NPY = ART_DIR / "sbert_embeddings.npy"        # (re)gerado pelo embedder.py
FAISS_INDEX = ART_DIR / "faiss_sbert.index"
MAPPING_PARQUET = ART_DIR / "doc_mapping.parquet" # id/title/text/lang/url alinhados

# TF-IDF (opcional; se existirem, usamos como reranker simples)
TFIDF_VECT = ART_DIR / "tfidf_vectorizer.pkl"
TFIDF_MAT  = ART_DIR / "tfidf_matrix.npz"

# LLM (Ollama)
OLLAMA_MODEL = "phi4-mini:latest"  # você já baixou

# Recuperação
TOP_K = 8
MAX_CONTEXT_TOKENS = 1600
