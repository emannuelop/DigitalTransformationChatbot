# chatbot/ui/views_ingest.py
from __future__ import annotations
import time
import streamlit as st
import requests
import os
import json

from .styles import inject_base_css

# Pega a URL da API do Docker Compose (ou localhost)
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

def ingest_screen(user: dict):
    # --- CORREÇÃO AQUI ---
    # A linha abaixo foi removida (comentada) porque 'app.py' já chama 'inject_base_css()'.
    # Chamar duas vezes causa o erro 'StreamlitDuplicateElementKey'.
    # inject_base_css()
    # --- FIM DA CORREÇÃO ---

    st.subheader("Adicionar base de conhecimento")
    st.caption(
        "Cole **links diretos de PDF** (terminados em **`.pdf`**) *Em português*, "
        "um por linha. Você também pode enviar arquivos **PDF** pelo upload."
    )

    # --- Widgets de Input ---
    urls_text = st.text_area(
        "Links de PDF (.pdf) — um por linha",
        placeholder="https://exemplo.gov.br/relatorio.pdf\nhttps://portal.gov.br/documento.pdf",
        height=120,
    )
    pdf_files = st.file_uploader(
        "Enviar PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        help="Envie aqui seus arquivos PDF (conteúdo preferencialmente em português).",
    )

    col1, col2 = st.columns([1, 1])
    with col1:
        back = st.button("← Voltar ao chat", use_container_width=True)
    with col2:
        run = st.button("Processar e atualizar índice", type="primary", use_container_width=True)

    if back:
        st.session_state["page"] = "chat"
        st.rerun()

    # ----------------- Ação: processar novas fontes -----------------
    if run:
        urls = [u.strip() for u in (urls_text or "").splitlines() if u.strip()]
        
        # --- LÓGICA PARA ENVIAR OS PDFs PARA A API ---
        files_to_upload = []
        if pdf_files:
            for f in pdf_files:
                # O formato é (filename, file_bytes, mime_type)
                files_to_upload.append(("pdf_files", (f.name, f.read(), f.type)))

        user_id = user["id"]

        with st.status("Processando fontes…", expanded=True) as status:
            status.write("➊ Enviando tarefa de ingestão para o microserviço…")
            try:
                # --- LÓGICA DE REQUEST ATUALIZADA ---
                form_data = {
                    'user_id': (None, str(user_id)),
                    'url_list_json': (None, json.dumps(urls)),
                }

                response = requests.post(
                    f"{API_URL}/ingest",
                    data=form_data,
                    files=files_to_upload
                )
                
                if not response.ok:
                    status.update(label="Falha ao contatar API 😵", state="error")
                    st.error(f"Erro da API: {response.text}")
                    return

                result = response.json()

            except Exception as e:
                status.update(label="Falha no pipeline 😵", state="error")
                st.exception(e)
                return

            for m in result.get("messages", []):
                status.write(m)
            
            status.write("➋ Tarefa recebida! O índice será atualizado em background. (Você ainda não pode utilizar a fonte adiconada até o processo terminar.)")
            status.update(label="Tarefa de atualização enviada! ✅", state="complete")

        try:
            st.cache_resource.clear()
        except Exception:
            pass

        added = result.get("added_items", []) or []
        if added:
            st.success(f"{len(added)} fonte(s) adicionada(s). Você pode renomear ou excluir abaixo.")
            with st.expander("Itens adicionados nesta operação", expanded=True):
                for it in added:
                    st.markdown(
                        f"- **{it.get('display_name') or it.get('title') or it.get('url')}** \n  `{it.get('url')}`"
                    )

    st.markdown("---")

    # ----------------- Minhas fontes (CRUD) -----------------
    st.subheader("Minhas fontes")
    try:
        user_id = user["id"]
        sources_resp = requests.get(f"{API_URL}/sources/{user_id}")
        sources_resp.raise_for_status() 
        sources = sources_resp.json()

    # --- Dedup por doc_id (defensivo) ---
        unique_sources = []
        _seen = set()
        for s in sources:
            did = s.get("doc_id")
            if did in _seen:
                continue
            _seen.add(did)
            unique_sources.append(s)

    except Exception as e:
        st.error(f"Falha ao carregar fontes: {e}")
        return

    if not sources:
        st.caption("Nenhuma fonte adicionada ainda.")
        return

    st.markdown('<div id="sources-list">', unsafe_allow_html=True)

    for i, src in enumerate(unique_sources):
        key_suffix = f"{src['doc_id']}_{i}"

        with st.container(border=True):
            left, right = st.columns([0.58, 0.42], vertical_alignment="center")

            with left:
                st.markdown(f'<div class="src-name">{src["display_name"]}</div>', unsafe_allow_html=True)
                st.markdown(
                    f'<div class="src-url"><a href="{src["url"]}" target="_blank">{src["url"]}</a></div>',
                    unsafe_allow_html=True,
                )

            with right:
                new_name = st.text_input(
                    "Renomear",
                    value=src["display_name"],
                    key=f"nm_{key_suffix}",           # <-- chave única
                    placeholder="Renomear",
                    label_visibility="collapsed",
                )
                c1, c2 = st.columns([0.5, 0.5])
                status_ph = st.empty()

                with c1:
                    if st.button("Salvar nome", key=f"save_{key_suffix}", use_container_width=True):   # <-- chave única
                        try:
                            status_ph.info("Renomeando…")
                            r_rename = requests.post(f"{API_URL}/rename", json={
                                "user_id": user["id"],
                                "doc_id": int(src["doc_id"]),
                                "new_name": new_name,
                            })
                            r_rename.raise_for_status()

                            status_ph.success("Nome atualizado ✅")
                            time.sleep(0.35)
                            st.rerun()

                        except Exception as e:
                            status_ph.error(f"Falha ao renomear: {e}")

                with c2:
                    if st.button("Excluir", key=f"del_{key_suffix}", use_container_width=True):        # <-- chave única
                        try:
                            status_ph.info("Excluindo e atualizando índice (em background)...")
                            r_delete = requests.post(f"{API_URL}/delete", json={
                                "user_id": user["id"],
                                "doc_id": int(src["doc_id"]),
                                "reindex": True,
                            })
                            r_delete.raise_for_status()

                            try:
                                st.cache_resource.clear()
                            except Exception:
                                pass

                            status_ph.success("Removido ✅")
                            time.sleep(0.45)
                            st.rerun()

                        except Exception as e:
                            status_ph.error(f"Falha ao excluir: {e}")

    st.markdown("</div>", unsafe_allow_html=True)

