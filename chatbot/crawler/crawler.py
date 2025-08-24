import argparse
import os
import re
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List
from urllib.parse import urljoin, urlparse, urlsplit, urlunsplit, parse_qsl, urlencode

import requests
from bs4 import BeautifulSoup, Comment
from pypdf import PdfReader
from requests.adapters import HTTPAdapter, Retry

# ==========================
# Constantes e diretórios
# ==========================
DATA_DIR = Path("data")
SEEDS_DIR = Path("seeds")
DATA_DIR.mkdir(exist_ok=True)
SEEDS_DIR.mkdir(exist_ok=True)

DB_PATH = os.environ.get("DB_PATH", str(DATA_DIR / "knowledge_base.db"))
SEEDS_FILE = os.environ.get("SEEDS_FILE", str(SEEDS_DIR / "seeds.txt"))

USER_AGENT = "Mozilla/5.0 (compatible; TCC-Crawler/1.0; +https://example.com/bot)"
HEADERS = {"User-Agent": USER_AGENT}
RATE_DELAY = 1.0              # segundos entre requisições
GET_TIMEOUT = 20              # timeout de GET
HEAD_TIMEOUT = 15             # timeout de HEAD

PDF_MIME_HINTS = ("application/pdf", "application/x-pdf", "binary/octet-stream")

# ignorar binários que não vamos extrair texto
SKIP_BINARIES_RX = re.compile(
    r"\.(jpg|jpeg|png|gif|bmp|svg|zip|rar|tar|gz|7z|mp4|mp3|pptx?|docx?)($|\?)",
    re.IGNORECASE,
)

# --------------------------
# Filtro de relevância suave
# --------------------------
TERMS = [
    # PT
    r"transforma[çc][aã]o digital",
    r"governo digital",
    r"estrat[eé]gia (de )?transforma[çc][aã]o digital",
    r"setor p[úu]blico",
    r"servi[çc]os digitais",
    r"gov\.br",
    # EN
    r"digital transformation",
    r"digital government",
    r"public sector",
    r"e-government",
    r"digital strategy",
]
TERMS_RX = [re.compile(t, re.IGNORECASE) for t in TERMS]

def soft_score(text: str) -> int:
    if not text:
        return 0
    return sum(1 for rx in TERMS_RX if rx.search(text))

def word_count(text: str) -> int:
    return len(re.findall(r"\w+", text or ""))

# ==========================
# URL helpers
# ==========================
def normalize_url(url: str) -> str:
    """Remove fragmentos e parâmetros de tracking comuns para evitar duplicatas 'disfarçadas'."""
    if not url:
        return url
    parts = urlsplit(url.strip())
    # fragmento (#...)
    parts = parts._replace(fragment="")
    # limpa tracking
    block = {"fbclid", "gclid"}
    qs = [(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True)
          if not k.lower().startswith("utm_") and k.lower() not in block]
    parts = parts._replace(query=urlencode(qs, doseq=True))
    # normaliza esquema/host para minúsculas
    scheme = (parts.scheme or "").lower()
    netloc = (parts.netloc or "").lower()
    parts = parts._replace(scheme=scheme, netloc=netloc)
    return urlunsplit(parts)

# ==========================
# Utilidades HTTP/DB/HTML
# ==========================
def make_session() -> requests.Session:
    session = requests.Session()
    retries = Retry(
        total=3,
        backoff_factor=0.8,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET", "HEAD"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.headers.update(HEADERS)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session

def init_db(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            url TEXT UNIQUE,
            content TEXT,
            content_type TEXT,
            fetched_at TEXT
        )
        """
    )
    # índice explícito ajuda nas consultas futuras
    cur.execute("CREATE INDEX IF NOT EXISTS idx_documents_url ON documents(url)")
    conn.commit()
    return conn

def load_seeds(path: str) -> List[str]:
    if not os.path.exists(path):
        return []
    seeds: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = normalize_url(line.strip())
            if not line or line.startswith("#"):
                continue
            seeds.append(line)
    return seeds

def is_pdf_suffix(url: str) -> bool:
    return url.lower().endswith(".pdf")

def head_content_type(session: requests.Session, url: str) -> str:
    try:
        r = session.head(url, timeout=HEAD_TIMEOUT, allow_redirects=True)
        return (r.headers.get("Content-Type") or "").lower()
    except Exception:
        return ""

def http_get(session: requests.Session, url: str, stream: bool = False) -> requests.Response:
    r = session.get(url, timeout=GET_TIMEOUT, stream=stream, allow_redirects=True)
    r.raise_for_status()
    return r

def visible_text_from_html(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "aside"]):
        tag.decompose()
    for element in soup.find_all(string=lambda s: isinstance(s, Comment)):
        element.extract()
    parts: List[str] = []
    for s in soup.find_all(string=True):
        parent = getattr(s, "parent", None)
        if parent and parent.name in {"style", "script", "head", "title", "meta", "[document]"}:
            continue
        st = s.strip()
        if st:
            parts.append(st)
    return "\n".join(parts)

def extract_pdf_text(raw: bytes) -> str:
    from io import BytesIO
    try:
        with BytesIO(raw) as bio:
            reader = PdfReader(bio)
            chunks: List[str] = []
            for page in reader.pages:
                try:
                    chunks.append(page.extract_text() or "")
                except Exception:
                    continue
            return "\n".join(chunks).strip()
    except Exception:
        return ""

def save_document(conn: sqlite3.Connection, url: str, content: str, content_type: str) -> None:
    cur = conn.cursor()
    cur.execute(
        """
        INSERT OR REPLACE INTO documents (url, content, content_type, fetched_at)
        VALUES (?,?,?,?)
        """,
        (url, content, content_type, datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()

def should_enqueue(link: str, base_host: str, same_domain: bool) -> bool:
    link = normalize_url(link)
    if not link.startswith(("http://", "https://")):
        return False
    if SKIP_BINARIES_RX.search(link):
        return False
    host = urlparse(link).hostname or ""
    return (host == base_host) if same_domain else True

# ==========================
# Núcleo
# ==========================
def process_url_and_discover(
    session: requests.Session,
    conn: sqlite3.Connection,
    url: str,
    same_domain: bool,
    min_score: int,
    min_words: int,
) -> List[str]:
    url = normalize_url(url)
    base_host = urlparse(url).hostname or ""

    # 1) PDF (sufixo)
    if is_pdf_suffix(url):
        resp = http_get(session, url, stream=True)
        content = extract_pdf_text(resp.content)
        if content and soft_score(content) >= min_score and word_count(content) >= min_words:
            save_document(conn, url, content, "pdf")
            print(f"[OK] PDF salvo: {url}")
        else:
            print(f"[SKIP] PDF filtrado: {url}")
        return []

    # 2) PDF (HEAD)
    ct = head_content_type(session, url)
    if any(hint in ct for hint in PDF_MIME_HINTS):
        resp = http_get(session, url, stream=True)
        content = extract_pdf_text(resp.content)
        if content and soft_score(content) >= min_score and word_count(content) >= min_words:
            save_document(conn, url, content, "pdf")
            print(f"[OK] PDF salvo (via HEAD): {url}")
        else:
            print(f"[SKIP] PDF filtrado (via HEAD): {url}")
        return []

    # 3) HTML
    resp = http_get(session, url, stream=False)
    html = resp.text
    text = visible_text_from_html(html)
    if soft_score(text) >= min_score and word_count(text) >= min_words:
        save_document(conn, url, text, "html")
        print(f"[OK] HTML salvo: {url}")
    else:
        print(f"[SKIP] HTML filtrado: {url}")

    soup = BeautifulSoup(html, "lxml")
    discovered: List[str] = []
    for a in soup.find_all("a", href=True):
        link = normalize_url(urljoin(url, a["href"]))
        if should_enqueue(link, base_host, same_domain):
            discovered.append(link)
    return list(dict.fromkeys(discovered))

def crawl(
    start_urls: Iterable[str],
    same_domain: bool = False,
    min_score: int = 1,
    min_words: int = 80,
    max_pages: int = 100,
    max_new_per_page: int = 1,
) -> None:
    conn = init_db(DB_PATH)
    session = make_session()

    seeds_q: List[str] = list(dict.fromkeys(normalize_url(u) for u in start_urls))
    disc_q: List[str] = []

    visited: set[str] = set()
    processed_discovered = 0

    print(
        f"Iniciando crawl | same_domain={same_domain} | "
        f"sementes={len(seeds_q)} | max_pages_descobertos={max_pages or '∞'} | "
        f"max_new_per_page={max_new_per_page} | min_score={min_score} | min_words={min_words}"
    )

    while seeds_q or disc_q:
        from_discovered = False
        if seeds_q:
            url = seeds_q.pop(0)
        else:
            if max_pages and processed_discovered >= max_pages:
                print(f"[STOP] Limite de descobertos atingido ({processed_discovered}/{max_pages}).")
                break
            url = disc_q.pop(0)
            from_discovered = True

        url = normalize_url(url)
        if url in visited:
            continue
        visited.add(url)

        try:
            new_links = process_url_and_discover(
                session=session,
                conn=conn,
                url=url,
                same_domain=same_domain,
                min_score=min_score,
                min_words=min_words,
            )

            if max_new_per_page > 0:
                new_links = new_links[:max_new_per_page]

            for link in new_links:
                if link not in visited and (link not in seeds_q) and (link not in disc_q):
                    disc_q.append(link)

        except requests.RequestException as e:
            print(f"[ERRO] Requisição falhou: {url} -> {e}")
        except Exception as e:
            print(f"[ERRO] Falha ao processar: {url} -> {e}")

        if from_discovered:
            processed_discovered += 1

        time.sleep(RATE_DELAY)

    conn.close()
    print(
        f"Concluído. Visitados: {len(visited)} | "
        f"Descobertos processados: {processed_discovered} | DB: {DB_PATH}"
    )

# ==========================
# CLI
# ==========================
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Crawler (sementes sempre; descobertos com limite leve) para HTML + PDF -> SQLite (data/knowledge_base.db)"
    )
    parser.add_argument(
        "--same-domain",
        type=lambda v: str(v).lower() in {"1", "true", "t", "yes", "y"},
        default=False,
        help="Se True, só segue links no mesmo domínio da página atual (padrão: False).",
    )
    parser.add_argument("--min-score", type=int, default=1,
                        help="Pontuação mínima de relevância (padrão: 1).")
    parser.add_argument("--min-words", type=int, default=80,
                        help="Mínimo de palavras no texto (padrão: 80).")
    parser.add_argument("--max-pages", type=int, default=100,
                        help="Teto de páginas processadas para LINKS DESCOBERTOS (0 = sem limite).")
    parser.add_argument("--max-new-per-page", type=int, default=1,
                        help="Máximo de novos links descobertos por página (padrão: 1).")

    args = parser.parse_args()
    seeds = load_seeds(SEEDS_FILE)
    if not seeds:
        print("Nenhuma semente encontrada em seeds/seeds.txt. Rode primeiro: python seeds/seeds_finder.py")
        return

    crawl(
        seeds,
        same_domain=args.same_domain,
        min_score=args.min_score,
        min_words=args.min_words,
        max_pages=args.max_pages,
        max_new_per_page=args.max_new_per_page,
    )

if __name__ == "__main__":
    main()