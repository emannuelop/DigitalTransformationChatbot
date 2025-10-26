# chatbot/ui/views_ingest.py
from __future__ import annotations
import time
import streamlit as st

from .styles import inject_base_css

def ingest_screen(user: dict):
    inject_base_css()

    st.subheader("Adicionar base de conhecimento")
    # 👉 Texto atualizado: somente links diretos de PDF em PT, terminados em .pdf
    st.caption(
        "Cole **links diretos de PDF** (terminados em **`.pdf`**) *Em português*, "
        "um por linha. Você também pode enviar arquivos **PDF** pelo upload."
    )

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
        uploads = []
        for f in pdf_files or []:
            uploads.append((f.name, f.read()))

        with st.status("Processando fontes…", expanded=True) as status:
            status.write("➊ Gravando uploads (se houver) e processando links…")
            try:
                from chatbot.ingestion import ingest_and_index
                result = ingest_and_index(url_list=urls, pdf_files=uploads)
            except Exception as e:
                status.update(label="Falha no pipeline 😵", state="error")
                st.exception(e)
                return

            for m in result.get("messages", []):
                status.write(m)
            status.write("➋ Atualizando embeddings e índice vetorial…")
            status.update(label="Índice atualizado com sucesso! ✅", state="complete")

        # Limpa caches para o chat usar o índice novo
        try:
            st.cache_resource.clear()
        except Exception:
            pass

        # Destaque: itens adicionados agora
        added = result.get("added_items", []) or []
        if added:
            st.success(f"{len(added)} fonte(s) adicionada(s). Você pode renomear ou excluir abaixo.")
            with st.expander("Itens adicionados nesta operação", expanded=True):
                for it in added:
                    st.markdown(
                        f"- **{it.get('display_name') or it.get('title') or it.get('url')}**  \n  `{it.get('url')}`"
                    )

    st.markdown("---")

    # ----------------- Minhas fontes (CRUD) -----------------
    st.subheader("Minhas fontes")
    try:
        from chatbot.ingestion import list_sources, rename_source, delete_source
        sources = list_sources()
    except Exception as e:
        st.error(f"Falha ao carregar fontes: {e}")
        return

    if not sources:
        st.caption("Nenhuma fonte adicionada ainda.")
        return

    # Wrapper para aplicar CSS específico desta lista
    st.markdown('<div id="sources-list">', unsafe_allow_html=True)

    for src in sources:
        # Card (retângulo) nativo do Streamlit
        with st.container(border=True):
            # Grid: nome/link (esq) | ações (dir)
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
                    key=f"nm_{src['doc_id']}",
                    placeholder="Renomear",
                    label_visibility="collapsed",
                )
                c1, c2 = st.columns([0.5, 0.5])
                status_ph = st.empty()  # feedback inline (loading / sucesso / erro)

                with c1:
                    if st.button("Salvar nome", key=f"save_{src['doc_id']}", use_container_width=True):
                        try:
                            status_ph.info("Renomeando…")
                            rename_source(int(src["doc_id"]), new_name)
                            status_ph.success("Nome atualizado ✅")
                            time.sleep(0.35)
                            st.rerun()
                        except Exception as e:
                            status_ph.error(f"Falha ao renomear: {e}")

                with c2:
                    if st.button("Excluir", key=f"del_{src['doc_id']}", use_container_width=True):
                        try:
                            status_ph.info("Excluindo e atualizando índice…")
                            delete_source(int(src["doc_id"]), reindex=True)
                            # Limpa caches para refletir imediatamente no chat
                            try:
                                st.cache_resource.clear()
                            except Exception:
                                pass
                            status_ph.success("Removido ✅")
                            time.sleep(0.45)
                            st.rerun()
                        except Exception as e:
                            status_ph.error(f"Falha ao excluir: {e}")

    st.markdown("</div>", unsafe_allow_html=True)  # fecha #sources-list
