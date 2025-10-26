# chatbot/extraction/user_ingest.py
from __future__ import annotations
from typing import Iterable, Tuple, Dict, Any, List
import sqlite3
import time
from pathlib import Path

from extraction.scraping import (
    DB_PATH, init_db, normalize_url,
    make_session, is_probably_html, fetch_html, extract_pdf_links,
    ensure_pdf_url, http_get_bytes, extract_pdf,
    word_count, soft_score, save_document, doc_exists,
    RATE_DELAY,
)

# --------------------------------------------------------------------
# Tabela auxiliar para gerenciar "Minhas fontes" (renomear/excluir)
# --------------------------------------------------------------------

def _ensure_user_sources_table(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS user_sources (
            doc_id       INTEGER PRIMARY KEY,
            display_name TEXT NOT NULL,
            created_at   DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(doc_id) REFERENCES documents(id) ON DELETE CASCADE
        )
    """)
    conn.commit()

def _get_doc_id_by_url(conn: sqlite3.Connection, url: str) -> int | None:
    row = conn.execute("SELECT id FROM documents WHERE url = ? ORDER BY id DESC LIMIT 1", (url,)).fetchone()
    return int(row[0]) if row else None

def _upsert_user_source(conn: sqlite3.Connection, doc_id: int, display_name: str) -> None:
    _ensure_user_sources_table(conn)
    conn.execute("""
        INSERT INTO user_sources (doc_id, display_name)
        VALUES (?, ?)
        ON CONFLICT(doc_id) DO UPDATE SET display_name=excluded.display_name
    """, (doc_id, display_name))
    conn.commit()

def _columns_in_table(conn: sqlite3.Connection, table: str) -> set[str]:
    cols = set()
    for row in conn.execute(f"PRAGMA table_info({table})").fetchall():
        cols.add(row[1])
    return cols

def list_user_sources() -> List[Dict[str, Any]]:
    """
    Retorna a lista de fontes adicionadas via UI (mapeadas em user_sources),
    de forma compatível com esquemas diferentes da tabela documents.
    Se 'wc' ou 'num_pages' não existirem no schema, retornamos None nesses campos.
    """
    conn = init_db(DB_PATH)
    _ensure_user_sources_table(conn)

    dcols = _columns_in_table(conn, "documents")
    select_bits = ["us.doc_id", "us.display_name", "d.url", "d.title"]

    # Inclui colunas opcionais somente se existirem
    if "num_pages" in dcols:
        select_bits.append("d.num_pages")
    else:
        select_bits.append("NULL AS num_pages")

    if "wc" in dcols:
        select_bits.append("d.wc")
    else:
        select_bits.append("NULL AS wc")

    select_sql = ", ".join(select_bits)
    rows = conn.execute(f"""
        SELECT {select_sql}, us.created_at
        FROM user_sources us
        JOIN documents d ON d.id = us.doc_id
        ORDER BY us.created_at DESC, us.doc_id DESC
    """).fetchall()
    conn.close()

    out = []
    for r in rows:
        # Ordem: doc_id, display_name, url, title, num_pages, wc, created_at
        out.append({
            "doc_id": int(r[0]),
            "display_name": r[1],
            "url": r[2],
            "title": r[3],
            "num_pages": r[4],
            "wc": r[5],
            "created_at": r[6],
        })
    return out

def rename_user_source(doc_id: int, new_name: str) -> None:
    conn = init_db(DB_PATH)
    _ensure_user_sources_table(conn)
    conn.execute("UPDATE user_sources SET display_name=? WHERE doc_id=?", (new_name.strip(), int(doc_id)))
    conn.commit()
    conn.close()

def delete_user_source(doc_id: int) -> None:
    """
    Remove a fonte do usuário e o documento bruto correspondente.
    (Depois do delete, rode processor->embedder->build_index para refletir no índice.)
    """
    conn = init_db(DB_PATH)
    _ensure_user_sources_table(conn)
    conn.execute("DELETE FROM user_sources WHERE doc_id=?", (int(doc_id),))
    conn.execute("DELETE FROM documents WHERE id=?", (int(doc_id),))
    conn.commit()
    conn.close()

# --------------------------------------------------------------------
# Ingestão SEM filtros (upload/URL) — grava SÓ o conteúdo no DB bruto
# --------------------------------------------------------------------

def _save_pdf_bytes(conn, url: str, raw: bytes, msgs: list[str], display_name: str) -> Dict[str, Any] | None:
    """
    Extrai texto e salva SEM filtros (sem MIN_WORDS/MIN_SCORE).
    Também registra a fonte em user_sources para permitir renomear/excluir.
    Retorna um dict com os metadados inseridos.
    """
    text, num_pages, title = extract_pdf(raw)
    if not text:
        msgs.append(f"[SKIP] Sem texto extraível: {url}")
        return None

    wc = word_count(text)
    sscore = soft_score(text)

    # Persiste conteúdo bruto
    save_document(conn, url, title, num_pages, wc, sscore, text)

    # Mapeia em user_sources (para CRUD na UI)
    doc_id = _get_doc_id_by_url(conn, url)
    if doc_id is not None:
        _upsert_user_source(conn, doc_id, display_name=display_name or (title or url))

    msgs.append(f"[OK] Salvo: {url} (pgs={num_pages} | wc={wc} | score={sscore})")
    return {"doc_id": doc_id, "display_name": display_name or (title or url), "url": url, "title": title}

def ingest_uploaded_pdfs_unfiltered(pdfs: Iterable[Tuple[str, bytes]] | None) -> Dict[str, Any]:
    """
    Recebe uploads [(nome, bytes)], salva no DB BRUTO SEM filtros e registra em user_sources.
    NÃO salva o PDF em disco — apenas o conteúdo (texto) no banco.
    """
    msgs: list[str] = []
    added = 0
    skipped = 0
    added_items: List[Dict[str, Any]] = []

    if not pdfs:
        return {"messages": [], "added": 0, "skipped": 0, "items": []}

    conn = init_db(DB_PATH)
    for name, raw in pdfs:
        url = normalize_url(f"upload://{name}")
        try:
            inserted = _save_pdf_bytes(conn, url, raw, msgs, display_name=name)
            if inserted:
                added_items.append(inserted)
                added += 1
            else:
                skipped += 1
        except Exception as e:
            skipped += 1
            msgs.append(f"[ERRO] Upload {name}: {e}")
    conn.close()
    return {"messages": msgs, "added": added, "skipped": skipped, "items": added_items}

def ingest_urls_unfiltered(urls: Iterable[str] | None) -> Dict[str, Any]:
    """
    Recebe URLs do usuário e salva PDFs SEM filtros de host/termos/word_count.
    - Se for página HTML, captura TODOS os links .pdf e salva.
    - Se for link direto para PDF, baixa e salva.
    Registra tudo em user_sources para gerenciamento na UI.
    """
    msgs: list[str] = []
    added = 0
    skipped = 0
    added_items: List[Dict[str, Any]] = []

    urls = [normalize_url((u or "").strip()) for u in (urls or [])]
    urls = [u for u in urls if u]  # sem allowed_host aqui (SEM filtro)

    if not urls:
        return {"messages": ["Nenhuma URL informada."], "added": 0, "skipped": 0, "items": []}

    conn = init_db(DB_PATH)
    session = make_session()
    unique = list(dict.fromkeys(urls))
    msgs.append(f"Ingestão sem filtros. URLs únicas: {len(unique)} | DB: {DB_PATH}")

    for u in unique:
        try:
            if is_probably_html(session, u):
                try:
                    html = fetch_html(session, u)
                except Exception as e:
                    msgs.append(f"[WARN] Falha ao abrir página: {u} -> {e}")
                    continue

                pdfs = extract_pdf_links(u, html)  # NÃO filtra host
                if not pdfs:
                    msgs.append(f"[INFO] Página sem links de PDF: {u}")
                    continue

                for pdf_url in pdfs:
                    if doc_exists(conn, pdf_url):
                        msgs.append(f"[SKIP] Já no banco: {pdf_url}")
                        continue
                    try:
                        if not ensure_pdf_url(session, pdf_url):
                            msgs.append(f"[SKIP] Não parece PDF: {pdf_url}")
                            continue
                        raw = http_get_bytes(session, pdf_url)
                        inserted = _save_pdf_bytes(
                            conn, pdf_url, raw, msgs,
                            display_name=Path(pdf_url).name or "arquivo.pdf",
                        )
                        if inserted:
                            added_items.append(inserted)
                            added += 1
                        else:
                            skipped += 1
                        time.sleep(RATE_DELAY)
                    except Exception as e:
                        skipped += 1
                        msgs.append(f"[ERRO] Baixar PDF {pdf_url}: {e}")
            else:
                # Link direto para um PDF
                if doc_exists(conn, u):
                    msgs.append(f"[SKIP] Já no banco: {u}")
                    continue
                if not ensure_pdf_url(session, u):
                    msgs.append(f"[SKIP] Não parece PDF: {u}")
                    continue
                try:
                    raw = http_get_bytes(session, u)
                    inserted = _save_pdf_bytes(
                        conn, u, raw, msgs,
                        display_name=Path(u).name or "arquivo.pdf",
                    )
                    if inserted:
                        added_items.append(inserted)
                        added += 1
                    else:
                        skipped += 1
                except Exception as e:
                    skipped += 1
                    msgs.append(f"[ERRO] Baixar PDF {u}: {e}")
                time.sleep(RATE_DELAY)
        except Exception as e:
            skipped += 1
            msgs.append(f"[ERRO] Ingestão URL {u}: {e}")

    conn.close()
    msgs.append("Concluído (sem filtros).")
    return {"messages": msgs, "added": added, "skipped": skipped, "items": added_items}
