# chatbot/ingestion.py
from __future__ import annotations
from typing import Dict, Any, List
from pathlib import Path
import sys
import time
import sqlite3  # <— para limpar o DB processado de imediato

# Aponta o Python para a pasta "chatbot"
REPO_ROOT = Path(__file__).resolve().parent  # ./chatbot
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# --- Ingestão de usuário (sem filtros) + CRUD de fontes ---
from extraction.user_ingest import (
    ingest_uploaded_pdfs_unfiltered,
    ingest_urls_unfiltered,
    list_user_sources,
    rename_user_source,
    delete_user_source,
)

# --- Pipeline ---
from ml import embedder as _embedder
from ml import build_index as _build_index
from ml.settings import ART_DIR
from clean.processor import run as _process  # seu processor local

# Tenta reutilizar os caminhos do processor; se falhar, calcula por fallback
try:
    from clean.processor import PROC_DB as _PROC_DB, RAW_DB as _RAW_DB
    PROC_DB = Path(_PROC_DB)
    RAW_DB = Path(_RAW_DB)
except Exception:
    PROC_DB = REPO_ROOT / "data" / "knowledge_base_processed.db"
    RAW_DB  = REPO_ROOT / "extraction" / "data" / "knowledge_base.db"

def _run_full_pipeline() -> None:
    print("[ingestion] ▶ processor.run()")
    _process()               # gera knowledge_base_processed.db etc.
    print("[ingestion] ▶ ml.embedder.main()")
    _embedder.main()         # sbert_mapping.parquet + sbert_embeddings.npy
    print("[ingestion] ▶ ml.build_index.main()")
    _build_index.main()      # faiss.index

# --------- Limpezas no DB processado (resolver “órfãos”) ---------

def _purge_processed_for(doc_id: int) -> None:
    """Remove do knowledge_base_processed.db tudo que referencie o original_id indicado."""
    if not PROC_DB.exists():
        return
    with sqlite3.connect(str(PROC_DB)) as conn:
        cur = conn.cursor()
        cur.execute("DELETE FROM processed_documents WHERE original_id = ?", (int(doc_id),))
        # duplicates_map pode conter referência cruzada — limpamos ambas as colunas
        cur.execute("DELETE FROM duplicates_map WHERE original_id = ? OR kept_original_id = ?", (int(doc_id), int(doc_id)))
        conn.commit()

def _purge_orphans_in_processed() -> None:
    """
    Remove do processado quaisquer registros cujo original_id NÃO exista mais no DB bruto.
    É um ‘safe-guard’ extra (útil se algo foi mexido fora do app).
    """
    if not PROC_DB.exists() or not RAW_DB.exists():
        return
    with sqlite3.connect(str(PROC_DB)) as proc, sqlite3.connect(str(RAW_DB)) as raw:
        proc_cur = proc.cursor()
        raw_cur  = raw.cursor()
        # ids existentes no bruto
        raw_ids = {row[0] for row in raw_cur.execute("SELECT id FROM documents")}
        # ids no processado
        proc_ids = [row[0] for row in proc_cur.execute("SELECT original_id FROM processed_documents")]
        to_drop = [oid for oid in proc_ids if oid not in raw_ids]
        if to_drop:
            proc_cur.executemany("DELETE FROM processed_documents WHERE original_id = ?", ((i,) for i in to_drop))
            # limpa também possíveis referências na duplicates_map
            proc_cur.executemany("DELETE FROM duplicates_map WHERE original_id = ? OR kept_original_id = ?",
                                 ((i, i) for i in to_drop))
            proc.commit()

def ingest_and_index(user_id: int, url_list=None, pdf_files=None) -> Dict[str, Any]:
    """
    1) adiciona PDFs/URLs SEM FILTROS ao DB bruto (e registra em user_sources)
    2) roda processor -> embedder -> build_index
    Retorna também a lista dos itens adicionados para a UI mostrar/gerir.
    """
    t0 = time.time()
    out_msgs: List[str] = []
    added_items: List[Dict[str, Any]] = []

    up = ingest_uploaded_pdfs_unfiltered(user_id, pdf_files)
    out_msgs += up.get("messages", [])
    added_items += up.get("items", [])

    se = ingest_urls_unfiltered(user_id, url_list)
    out_msgs += se.get("messages", [])
    added_items += se.get("items", [])

    _run_full_pipeline()

    return {
        "ok": True,
        "took_s": round(time.time() - t0, 2),
        "messages": out_msgs,
        "added_items": added_items,
        "artifacts_dir": str(ART_DIR),
    }

# ----------------------- API para a UI (gerenciamento) -----------------------

def list_sources(user_id: int) -> List[Dict[str, Any]]:
    return list_user_sources(user_id)

def rename_source(user_id: int, doc_id: int, new_name: str) -> None:
    rename_user_source(user_id, doc_id, new_name)

def delete_source(user_id: int, doc_id: int, reindex: bool = True) -> None:
    """
    Exclui a fonte do usuário + documento bruto
    e também limpa o registro correspondente do DB processado.
    Se reindex=True, roda processor->embedder->build_index para refletir no índice.
    """
    delete_user_source(user_id, doc_id)
    _purge_processed_for(doc_id)      # <— limpa do knowledge_base_processed.db
    _purge_orphans_in_processed()     # <— garantia extra (se algo foi mexido fora)
    if reindex:
        _run_full_pipeline()
