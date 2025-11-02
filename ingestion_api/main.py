# ingestion_api/main.py
from __future__ import annotations
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
import sys
from pathlib import Path
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks
import json
import time

# --- TRUQUE: Adiciona a pasta 'chatbot' ao path do Python ---
# Isso permite que este script importe 'chatbot.ingestion' etc.
CHATBOT_DIR = (Path(__file__).resolve().parent.parent / "chatbot").as_posix()
if CHATBOT_DIR not in sys.path:
    sys.path.insert(0, CHATBOT_DIR)
# -----------------------------------------------------------

# --- Imports do seu código (ATUALIZADO) ---
from chatbot.ingestion import (
    list_sources,
    rename_source,
    delete_source,
    _run_full_pipeline,
    _purge_processed_for,     # <--- Importado para o /delete
    _purge_orphans_in_processed # <--- Importado para o /delete
)
from chatbot.extraction.user_ingest import (
    ingest_uploaded_pdfs_unfiltered,
    ingest_urls_unfiltered,
)
# --- Fim dos imports atualizados ---


app = FastAPI(title="Ingestion Microservice")

# --- Pydantic Models para os dados JSON ---
class RenameRequest(BaseModel):
    user_id: int
    doc_id: int
    new_name: str

class DeleteRequest(BaseModel):
    user_id: int
    doc_id: int
    reindex: bool = True

# --- Endpoints da API ---

@app.get("/")
def health_check():
    return {"status": "ok", "service": "ingestion_api"}

# --- ENDPOINT /ingest ATUALIZADO ---
@app.post("/ingest", summary="Ingere PDFs e URLs e re-indexa")
async def api_ingest_and_index(
    background_tasks: BackgroundTasks,
    user_id: int = Form(...),
    url_list_json: str = Form("[]"),
    pdf_files: List[UploadFile] = File(None) # <--- NOVO: Recebe os PDFs
) -> Dict[str, Any]:
    """
    Este endpoint inicia a ingestão e indexação EM BACKGROUND.
    Retorna imediatamente para não travar a UI.
    """
    
    urls = json.loads(url_list_json)
    
    # --- NOVO: Converte UploadFile do FastAPI para o formato (name, bytes) ---
    # que a sua função 'ingest_uploaded_pdfs_unfiltered' espera.
    pdf_tuples = []
    if pdf_files:
        for f in pdf_files:
            pdf_tuples.append((f.filename, await f.read()))
    # --- FIM DA ATUALIZAÇÃO ---

    t0 = time.time()
    out_msgs = []
    added_items = []

    # Passo 1: Ingestão rápida (adaptado de 'ingest_and_index')
    
    # Roda a ingestão de PDFs (se houver)
    up = ingest_uploaded_pdfs_unfiltered(user_id, pdf_tuples) # <--- ATUALIZADO
    out_msgs += up.get("messages", [])
    added_items += up.get("items", [])

    # Roda a ingestão de URLs (se houver)
    se = ingest_urls_unfiltered(user_id, urls)
    out_msgs += se.get("messages", [])
    added_items += se.get("items", [])

    # Passo 2: Rodar o pipeline pesado em background
    # (Só roda se algo foi realmente adicionado)
    if added_items:
        background_tasks.add_task(_run_full_pipeline)
        out_msgs.append("Processamento de índice iniciado em background.")
    else:
        out_msgs.append("Nenhum item novo para adicionar.")

    # Retorna IMEDIATAMENTE para a UI
    return {
        "ok": True,
        "took_s": round(time.time() - t0, 2),
        "messages": out_msgs,
        "added_items": added_items,
        "artifacts_dir": "N/A (running in background)",
    }
# --- FIM DO ENDPOINT /ingest ATUALIZADO ---


@app.get("/sources/{user_id}", summary="Lista as fontes de um usuário")
def api_list_sources(user_id: int) -> List[Dict[str, Any]]:
    return list_sources(user_id)

@app.post("/rename", summary="Renomeia uma fonte")
def api_rename_source(request: RenameRequest):
    try:
        rename_source(request.user_id, request.doc_id, request.new_name)
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}

@app.post("/delete", summary="Exclui uma fonte e re-indexa")
def api_delete_source(request: DeleteRequest, background_tasks: BackgroundTasks):
    """
    Exclui a fonte e dispara a re-indexação em background (se solicitado).
    """
    try:
        # Faz a exclusão + purgas (SEM reindex síncrono)
        delete_source(request.user_id, request.doc_id, reindex=False)

        # Se quiser reindexar, roda o pipeline UMA vez, em background
        if request.reindex:
            background_tasks.add_task(_run_full_pipeline)

        return {"ok": True, "message": "Remoção concluída. Reindexação (se solicitada) iniciada em background."}
    except Exception as e:
        return {"ok": False, "error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)