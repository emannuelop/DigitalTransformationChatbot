from __future__ import annotations
import streamlit as st
from streamlit.components.v1 import html as _html

def base_css(sidebar_hidden: bool = False) -> str:
    return f"""
    <style>
      /* ===== Tipografia ===== */
      html, body, [class^="css"] {{
        font-family: system-ui, -apple-system, Segoe UI, Roboto, "Helvetica Neue", Arial, "Noto Sans",
                     "Liberation Sans", sans-serif !important;
        letter-spacing: .01em;
      }}

      :root {{
        --pad: 0.75rem;
        --radius: 14px;
        --radius-lg: 18px;

        --surface: rgba(255,255,255,.06);
        --border: rgba(255,255,255,.10);
      }}

      .block-container {{
        padding-top: 2rem !important;
        padding-bottom: 1rem !important;
        max-width: 1200px !important;
      }}

      /* ===== Chat ===== */
      [data-testid="stChatMessageAvatar"] {{ display: none !important; }}

      /* Perguntas do usuário (bolha com borda leve) */
      [data-testid="stChatMessage"][data-testid="stChatMessage-user"] > div,
      [data-testid="stChatMessage"][class*="user"] > div {{
        border-radius: 12px;
        padding: 0.85rem 1rem;
        word-break: break-word;
        margin: 0.4rem 0;
        border: 1px solid var(--border);
        background: var(--surface);
      }}

      /* Respostas do assistente — sem borda, fundo transparente */
      [data-testid="stChatMessage"][data-testid="stChatMessage-assistant"] > div,
      [data-testid="stChatMessage"][class*="assistant"] > div {{
        border: none !important;
        background: transparent !important;
        padding: 0.85rem 1rem;
        word-break: break-word;
        margin: 0.4rem 0;
      }}

      /* ===== Entrada do chat ===== */
      textarea[aria-label="Pergunte algo sobre a base de conhecimento…"] {{
        line-height: 1.4 !important;
      }}

      /* ===== Sugestões (pílulas) ===== */
      .sugg {{ margin-top: .25rem; }}           /* aproxima das frases do hero */
      .sugg button[data-testid="baseButton-secondary"] {{
        border: 1px solid var(--border) !important;
        background: var(--surface) !important;
        border-radius: 9999px !important;
        padding: .45rem .8rem !important;
        text-align: left !important;
        white-space: normal !important;
      }}

      /* ===== Sidebar ===== */
      [data-testid="stSidebar"] > div:first-child {{
        display: flex !important;
        flex-direction: column !important;
        height: 100% !important;
        gap: .5rem;
      }}
      [data-testid="stSidebar"] .sidebar-head {{ display:none !important; }}

      /* Botão “Novo chat” */
      [data-testid="stSidebar"] button[kind="primary"] {{
        margin-top: 0 !important;
        font-weight: 600 !important;
        height: 42px !important;
      }}

      /* Subtítulo “Histórico” */
      [data-testid="stSidebar"] .sb-section {{
        margin:.4rem 0 .15rem;
        font-size:.86rem; opacity:.78;
      }}

      /* Lista de chats */
      section[data-testid="stSidebar"] button[kind="secondary"] {{
        max-width: 100% !important;
        min-height: 38px !important;
        height: 38px !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: flex-start !important;
        gap: .5rem !important;
        padding: 0 .6rem !important;
        overflow: hidden !important;
        border-radius: 10px !important;
      }}
      section[data-testid="stSidebar"] button[kind="secondary"] * {{
        overflow: hidden !important;
        text-overflow: ellipsis !important;
        white-space: nowrap !important;
      }}

      /* ===== Userbar (rodapé da sidebar) ===== */
      .sb-userbar {{
        margin-top: auto !important;
        padding: .6rem .25rem .5rem .25rem;
        border-top: 1px solid var(--border);
      }}
      .sb-avatar {{
        width: 30px; height: 30px; border-radius: 9999px;
        background: var(--surface);
        display:flex; align-items:center; justify-content:center;
        font-weight:700; font-size:.9rem; user-select:none;
      }}
      .sb-avatar-img, .sb-avatar-img-96 {{
        border-radius: 50% !important;
        object-fit: cover !important;
        display: block !important;
      }}
      .sb-avatar-img {{ width:30px !important; height:30px !important; }}
      .sb-avatar-img-96 {{ width:96px !important; height:96px !important; }}

      /* Nome completo no popover */
      .sb-userbar button[data-testid="stPopoverButton"],
      .sb-userbar button[data-testid="stPopover"] {{
        width: 100% !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: flex-start !important;
        gap: .5rem !important;
        padding: 0 .6rem !important;
        min-height: 36px !important;
        height: auto !important;
        border-radius: 10px !important;
        overflow: visible !important;
        white-space: normal !important;
      }}
      .sb-userbar button[data-testid="stPopoverButton"] p,
      .sb-userbar button[data-testid="stPopover"] p {{
        overflow: visible !important;
        text-overflow: clip !important;
        white-space: normal !important;
        line-height: 1.2 !important;
        margin: 0 !important;
      }}

      /* Popover: divisores compactos */
      div[data-testid="stPopoverContent"] hr {{
        margin-top: .25rem !important; margin-bottom: .25rem !important;
      }}

      /* Seta da sidebar sempre visível */
      [data-testid="stSidebar"] button[aria-label*="colaps"],
      [data-testid="stSidebar"] button[aria-label*="ocultar"],
      [data-testid="stSidebar"] button[aria-label*="esconder"],
      [data-testid="stSidebar"] button[aria-label*="collapse"],
      [data-testid="stSidebar"] button[aria-label*="expand"],
      [data-testid="stSidebar"] button[title*="ocultar"],
      [data-testid="stSidebar"] button[title*="esconder"],
      [data-testid="stSidebar"] button[title*="collapse"],
      [data-testid="stSidebar"] button[title*="expand"] {{
        opacity: 1 !important; visibility: visible !important; transition: none !important;
      }}

      /* Empty state (aproxima sugestões do texto) */
      .hero-wrap {{
        min-height: 0 !important;            /* remove espaço gigante */
        display: block !important;            /* evita centralização vertical */
        margin: .25rem 0 .25rem 0 !important; /* aproxima do próximo bloco */
      }}
      .hero {{ width: 100%; max-width: 860px; text-align: center; margin: 0 auto .25rem; }}
      .hero h1 {{ font-size: 2rem; line-height: 1.3; margin: 0 0 .45rem 0; }}
      .hero p {{ opacity: .75; margin: 0; }}

      /* Responsividade */
      @media (max-width: 1024px) {{
        .block-container {{ padding-left: .75rem; padding-right: .75rem; }}
      }}
      @media (max-width: 900px) {{
        .block-container {{ padding-left: .6rem; padding-right: .6rem; }}
        .hero h1 {{ font-size: 1.7rem; }}
      }}
      @media (max-width: 680px) {{
        section[data-testid="stSidebar"] button[kind="secondary"] {{
          height: 40px !important;
        }}
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
