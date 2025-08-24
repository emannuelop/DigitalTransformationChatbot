import argparse
import os
import random
import time
from pathlib import Path
from typing import Iterable, Tuple
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from ddgs import DDGS  # pip install ddgs

# =============================================================================
# Paths
# =============================================================================

SEEDS_DIR = Path(__file__).parent
SEEDS_FILE = os.environ.get("SEEDS_FILE", str(SEEDS_DIR / "seeds.txt"))

# =============================================================================
# Consultas padrão (separadas por idioma)
# =============================================================================

QUERIES_PT: list[str] = [
    '"transformação digital"',
    '"governo digital"',
    '"estratégia de transformação digital"',
    '"transformação digital setor público"',
    '"serviços digitais"',
    '"inovação tecnológica" transformação digital',
    '"transformação digital" filetype:pdf',
    '"setor público" transformação digital" filetype:pdf',
]

QUERIES_EN: list[str] = [
    '"digital transformation"',
    '"digital government"',
    '"digital transformation strategy"',
    '"digital transformation public sector"',
    '"digital services"',
    '"digital transformation" filetype:pdf',
    '"digital transformation strategy" filetype:pdf',
]

# =============================================================================
# Filtros leves (descartam lixo óbvio)
# =============================================================================

DEFAULT_BLOCKED_HOST_SUFFIXES: set[str] = {
    "pinterest.com", "br.pinterest.com",
    "facebook.com", "m.facebook.com", "web.facebook.com",
    "instagram.com", "x.com", "twitter.com", "t.co",
    "tiktok.com", "whatsapp.com",
    "linktr.ee", "bit.ly", "goo.gl", "tinyurl.com", "lnkd.in",
    "reddit.com", "www.reddit.com", "discord.com",
}

BLOCKED_EXTS: set[str] = {
    # (PDF não está aqui de propósito)
    ".xml", ".rss", ".atom", ".json",
    ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".svg", ".webp",
    ".mp4", ".mp3", ".avi", ".mov", ".mkv",
    ".zip", ".rar", ".7z", ".tar", ".gz",
    ".ppt", ".pptx", ".doc", ".docx", ".xls", ".xlsx",
}

BAD_PATH_PARTS: set[str] = {
    "/sitemap", "/sitemap.xml", "/robots.txt", "/feed", "/rss", "/atom",
    "/wp-json", "/wp-login", "/login", "/signin", "/sign-in",
    "/tag/", "/tags/", "/category/", "/categoria/",
    "/search", "/pesquisa", "/?s=", "/?q=",
    "/calendar", "/evento", "/eventos",
    "/share", "/intent", "/send",
}

ACCENTED_CHARS = set("áàâãéêíóôõúçÁÀÂÃÉÊÍÓÔÕÚÇ")

# =============================================================================
# Utilidades
# =============================================================================


def normalize(url: str) -> str:
    """Remove fragmentos e tracking; normaliza esquema/host; remove '/' final redundante."""
    if not url:
        return url
    parts = urlsplit(url.strip())
    parts = parts._replace(fragment="")

    # remove parâmetros comuns de tracking
    blocked_params = {"fbclid", "gclid"}
    new_qs = [
        (k, v)
        for k, v in parse_qsl(parts.query, keep_blank_values=True)
        if not k.lower().startswith("utm_") and k.lower() not in blocked_params
    ]
    parts = parts._replace(query=urlencode(new_qs, doseq=True))

    # esquema e host em minúsculas
    parts = parts._replace(
        scheme=(parts.scheme or "").lower(),
        netloc=(parts.netloc or "").lower(),
    )

    cleaned = urlunsplit(parts)

    # remove barra final redundante (sem tocar na raiz)
    base = f"{parts.scheme}://{parts.netloc}"
    if cleaned.endswith("/") and len(cleaned) > len(base) + 1:
        cleaned = cleaned[:-1]
    return cleaned


def load_existing(path: str) -> set[str]:
    """Carrega seeds existentes, ignorando linhas vazias e comentários."""
    if not os.path.exists(path):
        return set()
    out: set[str] = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s and not s.startswith("#"):
                out.add(s)
    return out


def save_merged(path: str, new_items: set[str]) -> None:
    """Preserva comentários existentes e mescla URLs deduplicadas."""
    header: list[str] = []
    old_urls: list[str] = []

    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip() or line.startswith("#"):
                    header.append(line)
                else:
                    old_urls.append(line.strip())

    merged: list[str] = []
    seen: set[str] = set()

    # mantém ordem do arquivo atual
    for u in old_urls:
        if u not in seen:
            merged.append(u)
            seen.add(u)

    # adiciona novos (ordenados) ao final
    for u in sorted(new_items):
        if u not in seen:
            merged.append(u)
            seen.add(u)

    with open(path, "w", encoding="utf-8") as f:
        if header:
            f.writelines(header)
        for u in merged:
            f.write(u + "\n")


def looks_useful_url(url: str, extra_blocked: set[str]) -> bool:
    """Heurística barata para descartar URLs pouco úteis como seed."""
    parts = urlsplit(url)
    host = (parts.netloc or "").lower()
    if not host:
        return False

    # bloquear domínios/sufixos conhecidos por serem pouco úteis
    for suf in (DEFAULT_BLOCKED_HOST_SUFFIXES | extra_blocked):
        if host == suf or host.endswith("." + suf):
            return False

    # bloquear extensões não textuais
    p = parts.path.lower()
    for ext in BLOCKED_EXTS:
        if p.endswith(ext):
            return False

    # bloquear caminhos ruidosos
    low = (parts.path + "?" + parts.query).lower()
    if any(bad in low for bad in BAD_PATH_PARTS):
        return False

    # esquemas indesejados
    if url.startswith(("mailto:", "javascript:", "tel:")):
        return False

    return True


def ddg_search(query: str, lang: str, limit: int) -> list[str]:
    """Executa busca no DuckDuckGo (ddgs) e retorna URLs."""
    region = "br-pt" if lang == "pt" else "wt-wt"
    urls: list[str] = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, region=region, safesearch="moderate", max_results=limit):
            href = r.get("href") or r.get("link") or r.get("url")
            if href:
                urls.append(href)
    return urls[:limit]


def split_queries_from_file(path: Path) -> Tuple[list[str], list[str]]:
    """Separa queries por idioma com heurística simples (acentuação -> PT)."""
    queries_pt: list[str] = []
    queries_en: list[str] = []

    for raw in path.read_text(encoding="utf-8").splitlines():
        q = raw.strip()
        if not q or q.startswith("#"):
            continue
        if any(ch in q for ch in ACCENTED_CHARS):
            queries_pt.append(q)
        else:
            queries_en.append(q)

    return queries_pt, queries_en


def build_lang_query_sets(
    max_queries: int, queries_file: str | None
) -> list[Tuple[str, list[str]]]:
    """Monta a lista [(lang, queries)] respeitando max_queries e arquivo opcional."""
    if queries_file:
        qfile = Path(queries_file)
        if not qfile.exists():
            raise FileNotFoundError(f"Arquivo de queries não encontrado: {qfile}")
        queries_pt, queries_en = split_queries_from_file(qfile)
    else:
        queries_pt = QUERIES_PT
        queries_en = QUERIES_EN

    if max_queries and max_queries > 0:
        queries_pt = queries_pt[:max_queries]
        queries_en = queries_en[:max_queries]

    return [("pt", queries_pt), ("en", queries_en)]


# =============================================================================
# CLI
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(description="Gerador de seeds (DuckDuckGo/ddgs).")
    parser.add_argument("--top", type=int, default=15, help="Resultados por query.")
    parser.add_argument("--max-queries", type=int, default=0, help="Limite de queries por idioma (0 = todas).")
    parser.add_argument("--sleep", type=float, default=1.0, help="Intervalo base entre buscas (s).")
    parser.add_argument("--langs", choices=["pt", "en", "both"], default="both",
                        help="Quais idiomas rodar.")
    parser.add_argument("--queries-file", type=str, default="",
                        help="Arquivo .txt com uma query por linha (PT/EN separados por acentuação).")
    parser.add_argument("--block-domains", type=str, default="",
                        help="Domínios extra a bloquear (vírgula separada).")
    args = parser.parse_args()

    # queries por idioma
    try:
        lang_sets = build_lang_query_sets(args.max_queries, args.queries_file or None)
    except FileNotFoundError as e:
        print(f"[ERRO] {e}")
        return

    # filtra idiomas a executar
    if args.langs == "pt":
        lang_sets = [s for s in lang_sets if s[0] == "pt"]
    elif args.langs == "en":
        lang_sets = [s for s in lang_sets if s[0] == "en"]

    # blocklist extra
    extra_blocked = {
        d.strip().lower()
        for d in (args.block_domains.split(",") if args.block_domains else [])
        if d.strip()
    }

    existing = load_existing(SEEDS_FILE)
    collected_pt: set[str] = set()
    collected_en: set[str] = set()

    total_steps = sum(len(qs) for _, qs in lang_sets)
    step = 0

    for lang, queries in lang_sets:
        for q in queries:
            step += 1
            try:
                results = ddg_search(q, lang, args.top)
            except Exception as e:
                print(f"[WARN] Falha na busca '{q}' (lang={lang}): {e}")
                results = []

            added = 0
            for url in results:
                u = normalize(url)
                if not u:
                    continue
                if u in existing or u in collected_pt or u in collected_en:
                    continue
                if not looks_useful_url(u, extra_blocked=extra_blocked):
                    continue

                (collected_pt if lang == "pt" else collected_en).add(u)
                added += 1

            print(f"[OK] {step}/{total_steps} '{q}' [{lang}] -> +{added} links")
            time.sleep(args.sleep + random.uniform(0, 0.4))

    total_new = len(collected_pt) + len(collected_en)
    if total_new == 0:
        print("Nenhum novo link encontrado com os critérios atuais.")
        return

    save_merged(SEEDS_FILE, collected_pt | collected_en)

    print("\nResumo por idioma:")
    print(f"  PT: +{len(collected_pt)} links novos")
    print(f"  EN: +{len(collected_en)} links novos")
    print(f"\nTotal adicionado: {total_new} | arquivo: {SEEDS_FILE}")
    print("Obs.: o arquivo foi mesclado e deduplicado; URLs existentes foram preservadas.")


if __name__ == "__main__":
    main()
