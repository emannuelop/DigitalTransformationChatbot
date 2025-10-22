from __future__ import annotations
import streamlit as st
from streamlit.components.v1 import html as _html

def base_css(sidebar_hidden: bool = False) -> str:
    # Mantemos a seta nativa da sidebar (sem hacks JS)
    return f"""
    <style>
      :root {{
        --pad: 0.75rem;
      }}

      .block-container {{
        padding-top: 2.25rem;
        padding-bottom: 1rem;
        max-width: 1200px;
      }}

      /* ===== Chat ===== */
      [data-testid="stChatMessageAvatar"] {{ display: none !important; }}
      [data-testid="stChatMessage"] > div {{
        border-radius: 14px;
        padding: .75rem 1rem;
        word-break: break-word;
        margin-left: 0 !important;
      }}
      [data-testid="stChatMessage"]:first-of-type {{ margin-top: .75rem; }}

      /* ===== Sidebar shell (flex p/ colar o userbar no rodapé) ===== */
      [data-testid="stSidebar"] > div:first-child {{
        display: flex !important;
        flex-direction: column !important;
        height: 100% !important;
        gap: .5rem;
      }}

      /* Tira o título “Conversas” se existir em versões antigas */
      [data-testid="stSidebar"] .sidebar-head {{ display:none !important; }}

      /* Botão “Novo chat” mais no topo */
      [data-testid="stSidebar"] button[kind="primary"] {{
        margin-top: 0 !important;
      }}

      /* Seção de histórico (subtítulo discreto) */
      [data-testid="stSidebar"] .sb-section {{
        margin:.4rem 0 .2rem; font-size:.85rem; opacity:.75;
      }}

      /* Botões da lista de chats: truncar texto e não vazar */
      section[data-testid="stSidebar"] button[kind="secondary"] {{
        max-width: 100% !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
        white-space: nowrap !important;
        padding-right: .6rem !important;
      }}
      section[data-testid="stSidebar"] button[kind="secondary"] * {{
        overflow: hidden !important;
        text-overflow: ellipsis !important;
        white-space: nowrap !important;
      }}

      /* ===== Botão de popover (seta) SEMPRE visível e com tamanho mínimo ===== */
      [data-testid="stSidebar"] [data-testid="stPopoverButton"] {{
        width: 36px !important;
        min-width: 36px !important;
        height: 36px !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        opacity: 1 !important;
        visibility: visible !important;
      }}

      /* ===== Userbar fixo no rodapé (avatar + nome) ===== */
      .sb-userbar {{
        margin-top: auto !important;                 /* cola no rodapé */
        padding: .6rem .25rem .4rem .25rem;
        border-top: 1px solid rgba(255,255,255,.08);
      }}
      .sb-avatar {{
        width: 30px; height: 30px; border-radius: 9999px;
        background: rgba(255,255,255,.08);
        display:flex; align-items:center; justify-content:center;
        font-weight:700; font-size:.9rem; user-select:none;
      }}
      /* Foto redonda (30px e 96px) */
      .sb-avatar-img, .sb-avatar-img-96 {{
        border-radius: 50% !important;
        object-fit: cover !important;
        display: block !important;
      }}
      .sb-avatar-img {{ width:30px !important; height:30px !important; }}
      .sb-avatar-img-96 {{ width:96px !important; height:96px !important; }}

      /* Nome (botão do popover) sempre truncando */
      .sb-userbar button[data-testid="stPopoverButton"],
      .sb-userbar button[data-testid="stPopover"] {{
        max-width: 100% !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
        white-space: nowrap !important;
      }}
      .sb-userbar button[data-testid="stPopoverButton"] p,
      .sb-userbar button[data-testid="stPopover"] p {{
        max-width: 100% !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
        white-space: nowrap !important;
      }}

      /* Popover: divisores compactos */
      div[data-testid="stPopoverContent"] hr {{
        margin-top: .25rem !important;
        margin-bottom: .25rem !important;
      }}

      /* ===== Seta de recolher/expandir da SIDEBAR SEMPRE visível ===== */
      [data-testid="stSidebar"] button[aria-label*="colaps"],
      [data-testid="stSidebar"] button[aria-label*="ocultar"],
      [data-testid="stSidebar"] button[aria-label*="esconder"],
      [data-testid="stSidebar"] button[aria-label*="collapse"],
      [data-testid="stSidebar"] button[aria-label*="expand"],
      [data-testid="stSidebar"] button[title*="ocultar"],
      [data-testid="stSidebar"] button[title*="esconder"],
      [data-testid="stSidebar"] button[title*="collapse"],
      [data-testid="stSidebar"] button[title*="expand"] {{
        opacity: 1 !important;
        visibility: visible !important;
        transition: none !important;
      }}

      /* ===== Empty state ===== */
      .hero-wrap {{
        min-height: calc(100vh - 200px);
        display: flex; align-items: center; justify-content: center;
      }}
      .hero {{ width: 100%; max-width: 860px; text-align: center; margin: 0 auto; }}
      .hero h1 {{ font-size: 2rem; line-height: 1.3; margin: 0 0 .75rem 0; }}
      .hero p {{ opacity: .75; margin: 0 0 1rem 0; }}

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
