from __future__ import annotations
import streamlit as st
from streamlit.components.v1 import html as _html

def base_css(sidebar_hidden: bool = False) -> str:
    # Mantemos a seta nativa da sidebar (sem hacks)
    return f"""
    <style>
      :root {{
        --pad: 0.75rem;
      }}

      .block-container {{
        padding-top: 1rem;
        padding-bottom: 1rem;
        max-width: 1200px;
      }}

      /* ===== Chat ===== */
      /* Some avatar/emoji nas mensagens */
      [data-testid="stChatMessageAvatar"] {{
        display: none !important;
      }}
      /* Ajusta o padding quando não há avatar */
      [data-testid="stChatMessage"] > div {{
        border-radius: 14px;
        padding: .75rem 1rem;
        word-break: break-word;
        margin-left: 0 !important;
      }}

      /* ===== Sidebar ===== */
      [data-testid="stSidebar"] .sidebar-head {{
        display:flex; align-items:center; justify-content:space-between;
        font-weight:600; opacity:.9; margin-bottom:.4rem;
      }}
      [data-testid="stSidebar"] .sb-section {{
        margin:.4rem 0 .2rem; font-size:.85rem; opacity:.75;
      }}
      /* Botões com elipse e tooltip */
      button[kind="secondary"] p, button[kind="secondary"] span {{
        overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
      }}

      /* ===== Empty state (estilo ChatGPT) ===== */
      .hero-wrap {{
        min-height: calc(100vh - 200px);
        display: flex; align-items: center; justify-content: center;
      }}
      .hero {{
        width: 100%; max-width: 860px; text-align: center;
        margin: 0 auto;
      }}
      .hero h1 {{
        font-size: 2rem; line-height: 1.3; margin: 0 0 .75rem 0;
      }}
      .hero p {{
        opacity: .75; margin: 0 0 1rem 0;
      }}

      /* ===== Responsividade ===== */
      @media (max-width: 900px) {{
        .block-container {{ padding-left: .6rem; padding-right: .6rem; }}
        .hero h1 {{ font-size: 1.7rem; }}
      }}
    </style>
    """

def inject_base_css(sidebar_hidden: bool = False) -> None:
    st.markdown(base_css(sidebar_hidden), unsafe_allow_html=True)

def inject_hotkeys(enable: bool = False) -> None:
    # Opcional (não precisa com st.chat_input). Mantido caso queira ativar depois.
    if not enable:
        return
    _html(
        """
        <script>
        (function() {
          const root = window.parent.document;
          function isTextInput(el){
            if (!el) return false;
            const tag = el.tagName;
            return tag === 'INPUT' || tag === 'TEXTAREA' || el.getAttribute('contenteditable') === 'true';
          }
          function clickSend(){
            const btn = root.querySelector('button[aria-label="Send message"]');
            if (btn) btn.click();
          }
          root.addEventListener('keydown', function(e){
            const a = root.activeElement;
            if (!isTextInput(a)) return;
            if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); clickSend(); }
          }, true);
        })();
        </script>
        """,
        height=0,
        width=0,
    )
