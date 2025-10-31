import os
import re
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit, urljoin, parse_qsl, urlencode

import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader
from requests.adapters import HTTPAdapter, Retry

BASE_DIR  = Path(__file__).parent
DATA_DIR  = BASE_DIR / "data"
SEEDS_DIR = BASE_DIR / "seeds"
DATA_DIR.mkdir(exist_ok=True)
SEEDS_DIR.mkdir(exist_ok=True)

DB_PATH       = os.environ.get("DB_PATH", str(DATA_DIR / "knowledge_base.db"))
SEEDS_FILE_PT = SEEDS_DIR / "seeds_pt.txt" 

USER_AGENT  = "Mozilla/5.0 (compatible; TCC-Scraping/3.2; +https://example.com/bot)"
GET_TIMEOUT = 30          
CHUNK_SIZE  = 128 * 1024   
RATE_DELAY  = 1.0          
PDF_MAGIC   = b"%PDF-"     

MIN_WORDS = 80
MIN_SCORE = 1
TERMS = [
    r"transforma[çc][aã]o digital",
    r"governo digital",
    r"setor p[úu]blico",
    r"servi[çc]os p[úu]blicos digitais",
    r"gest[aã]o p[úu]blica",
    r"pol[ií]ticas p[úu]blicas",
    r"governan[çc]a digital",
    r"interoperabilidade",
    r"dados abertos",
    r"cidades inteligentes",
    r"sa[úu]de digital",
    r"educa[çc][aã]o digital",
    r"justi[çc]a digital",
    r"gov\.br",
]
TERMS_RX = [re.compile(t, re.IGNORECASE) for t in TERMS]

BLOCKED_HOST_SUFFIXES = {
    "pinterest.com", "br.pinterest.com",
    "facebook.com", "m.facebook.com", "web.facebook.com",
    "instagram.com", "x.com", "twitter.com", "t.co",
    "tiktok.com", "whatsapp.com",
    "linktr.ee", "bit.ly", "goo.gl", "tinyurl.com", "lnkd.in",
    "reddit.com", "www.reddit.com", "discord.com",
}
ALLOWED_SCHEMES = {"http", "https"}

def soft_score(text: str) -> int:
    return sum(1 for rx in TERMS_RX if rx.search(text or ""))

def word_count(text: str) -> int:
    return len(re.findall(r"\w+", text or ""))

def normalize_url(url: str) -> str:
    if not url:
        return url
    parts = urlsplit(url.strip())
    scheme = (parts.scheme or "").lower()
    if scheme and scheme not in ALLOWED_SCHEMES:
        return ""
    parts = parts._replace(fragment="")
    blocked_params = {"fbclid", "gclid"}
    qs = [(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True)
          if not k.lower().startswith("utm_") and k.lower() not in blocked_params]
    parts = parts._replace(query=urlencode(qs, doseq=True))
    parts = parts._replace(scheme=scheme, netloc=(parts.netloc or "").lower())
    cleaned = urlunsplit(parts)
    base = f"{parts.scheme}://{parts.netloc}"
    if cleaned.endswith("/") and len(cleaned) > len(base) + 1:
        cleaned = cleaned[:-1]
    return cleaned

def allowed_host(url: str) -> bool:
    host = (urlsplit(url).netloc or "").lower()
    if not host:
        return False
    return not any(host == suf or host.endswith("." + suf) for suf in BLOCKED_HOST_SUFFIXES)

def is_pdf_like(url: str) -> bool:
    return ".pdf" in (urlsplit(url).path or "").lower()

def make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT})
    retries = Retry(
        total=3,
        backoff_factor=0.7,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET", "HEAD"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retries)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    return s

def ensure_table(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            url TEXT UNIQUE,
            title TEXT,
            num_pages INTEGER,
            word_count INTEGER,
            soft_score INTEGER,
            content TEXT,
            content_type TEXT,
            fetched_at TEXT,
            user_id INTEGER -- NULL para documentos globais (scraping), ID para documentos de usuário
        )
        """
    )
    cur.execute("CREATE INDEX IF NOT EXISTS idx_documents_url ON documents(url)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_documents_user_id ON documents(user_id)")
    conn.commit()

def init_db(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    ensure_table(conn)
    return conn

def doc_exists(conn: sqlite3.Connection, url: str) -> bool:
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM documents WHERE url = ? LIMIT 1", (url,))
    return cur.fetchone() is not None

def save_document(conn: sqlite3.Connection, url: str, title: str,
                  num_pages: int, wc: int, sscore: int, content: str, user_id: int | None = None) -> None:
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO documents (url, title, num_pages, word_count, soft_score, content, content_type, fetched_at, user_id)
        VALUES (?, ?, ?, ?, ?, ?, 'pdf', ?, ?)
        ON CONFLICT(url) DO UPDATE SET
          title=excluded.title,
          num_pages=excluded.num_pages,
          word_count=excluded.word_count,
          soft_score=excluded.soft_score,
          content=excluded.content,
          content_type='pdf',
          fetched_at=excluded.fetched_at,
          user_id=excluded.user_id
        """,
        (url, title or None, num_pages, wc, sscore, content, datetime.now(timezone.utc).isoformat(), user_id),
    )
    conn.commit()

def head_is_pdf(session: requests.Session, url: str) -> bool:
    try:
        r = session.head(url, allow_redirects=True, timeout=GET_TIMEOUT)
        ct = (r.headers.get("Content-Type") or "").lower()
        return 200 <= r.status_code < 400 and ("application/pdf" in ct or "octet-stream" in ct)
    except Exception:
        return False

def peek_is_pdf(session: requests.Session, url: str, peek_bytes: int = 8192) -> bool:
    try:
        with session.get(url, stream=True, allow_redirects=True, timeout=GET_TIMEOUT) as r:
            if not (200 <= r.status_code < 400):
                return False
            chunk = next(r.iter_content(chunk_size=peek_bytes), b"")
            return chunk.startswith(PDF_MAGIC)
    except Exception:
        return False

def ensure_pdf_url(session: requests.Session, url: str) -> bool:
    if head_is_pdf(session, url):
        return True
    if is_pdf_like(url):
        return peek_is_pdf(session, url)
    return False

def http_get_bytes(session: requests.Session, url: str) -> bytes:
    r = session.get(url, timeout=GET_TIMEOUT, stream=True, allow_redirects=True)
    r.raise_for_status()
    chunks: list[bytes] = []
    for chunk in r.iter_content(chunk_size=CHUNK_SIZE):
        if chunk:
            chunks.append(chunk)
    return b"".join(chunks)

def extract_pdf(raw: bytes) -> tuple[str, int, str]:
    from io import BytesIO
    try:
        with BytesIO(raw) as bio:
            reader = PdfReader(bio)
            parts: list[str] = []
            for page in reader.pages:
                try:
                    parts.append(page.extract_text() or "")
                except Exception:
                    continue
            text = "\n".join(parts).strip()
            num_pages = len(reader.pages) if reader.pages else 0
            title = ""
            try:
                meta = getattr(reader, "metadata", None) or {}
                title = (meta.get("/Title") or meta.get("Title") or "").strip()
            except Exception:
                pass
            return text, num_pages, title
    except Exception:
        return "", 0, ""

def load_seeds_pt(path: Path) -> list[str]:
    if not path.exists():
        return []
    items: list[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            url = normalize_url(raw.strip())
            if url and not url.startswith("#") and allowed_host(url):
                items.append(url)
    return items

def is_probably_html(session: requests.Session, url: str) -> bool:
    if is_pdf_like(url):
        return False
    try:
        r = session.head(url, allow_redirects=True, timeout=GET_TIMEOUT)
        ct = (r.headers.get("Content-Type") or "").lower()
        if "text/html" in ct:
            return True
        if "application/pdf" in ct:
            return False
        with session.get(url, stream=True, allow_redirects=True, timeout=GET_TIMEOUT) as g:
            if not (200 <= g.status_code < 400):
                return False
            chunk = next(g.iter_content(chunk_size=2048), b"").lower()
            return b"<html" in chunk or b"<!doctype html" in chunk
    except Exception:
        return False

def fetch_html(session: requests.Session, url: str) -> str:
    r = session.get(url, timeout=GET_TIMEOUT, allow_redirects=True)
    r.raise_for_status()
    return r.text

def extract_pdf_links(base_url: str, html: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    candidates: list[str] = []
    for a in soup.find_all("a", href=True):
        abs_url = normalize_url(urljoin(base_url, a["href"]))
        if abs_url and allowed_host(abs_url) and is_pdf_like(abs_url):
            candidates.append(abs_url)
    return list(dict.fromkeys(candidates))

def process_pdf(session: requests.Session, conn: sqlite3.Connection, url: str) -> None:
    if not ensure_pdf_url(session, url):
        print(f"[SKIP] Não é PDF válido/baixável: {url}")
        return
    try:
        raw = http_get_bytes(session, url)
    except Exception as e:
        print(f"[ERRO] GET falhou: {url} -> {e}")
        return

    text, num_pages, title = extract_pdf(raw)
    if not text:
        print(f"[SKIP] PDF sem texto extraível: {url}")
        return

    wc = word_count(text)
    sscore = soft_score(text)
    if wc < MIN_WORDS or sscore < MIN_SCORE:
        print(f"[SKIP] PDF filtrado por conteúdo (wc={wc}, score={sscore}): {url}")
        return

    # Documentos do scraping são globais, user_id=None
    save_document(conn, url, title, num_pages, wc, sscore, text, user_id=None)
    print(f"[OK] Salvo (pgs={num_pages} | wc={wc} | score={sscore}): {url}")

def crawl_pdf_only_pt(seeds: list[str]) -> None:
    conn = init_db(DB_PATH)
    session = make_session()

    unique = list(dict.fromkeys(seeds))
    print(f"Iniciando (PDF-only, PT). Seeds únicas: {len(unique)} | DB: {DB_PATH}")

    for seed in unique:
        try:
            if is_probably_html(session, seed):
                try:
                    html = fetch_html(session, seed)
                except Exception as e:
                    print(f"[WARN] Falha ao abrir página seed: {seed} -> {e}")
                    continue
                for pdf_url in extract_pdf_links(seed, html):
                    if doc_exists(conn, pdf_url):
                        print(f"[SKIP] Já no banco: {pdf_url}")
                        continue
                    process_pdf(session, conn, pdf_url)
                    time.sleep(RATE_DELAY)
            else:
                if doc_exists(conn, seed):
                    print(f"[SKIP] Já no banco: {seed}")
                else:
                    process_pdf(session, conn, seed)
                time.sleep(RATE_DELAY)
        except Exception as e:
            print(f"[ERRO] Falha ao processar seed: {seed} -> {e}")

    conn.close()
    print("Concluído.")

def main() -> None:
    seeds_pt = load_seeds_pt(SEEDS_FILE_PT)
    if not seeds_pt:
        print("Nenhuma seed PT encontrada. Gere antes com: python seeds_finder.py")
        return
    crawl_pdf_only_pt(seeds_pt)

if __name__ == "__main__":
    main()
