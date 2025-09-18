import os
import re
import sqlite3
import time
import requests

from datetime import datetime, timezone
from pathlib import Path
from pypdf import PdfReader
from requests.adapters import HTTPAdapter, Retry

# ----------------------------
# Paths fixos
# ----------------------------
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
SEEDS_DIR = BASE_DIR / "seeds"
DATA_DIR.mkdir(exist_ok=True)
SEEDS_DIR.mkdir(exist_ok=True)

DB_PATH = os.environ.get("DB_PATH", str(DATA_DIR / "knowledge_base.db"))
SEEDS_FILE_PT = SEEDS_DIR / "seeds_pt.txt"
SEEDS_FILE_EN = SEEDS_DIR / "seeds_en.txt"

# ----------------------------
# Parâmetros
# ----------------------------
USER_AGENT = "Mozilla/5.0 (compatible; TCC-Crawler/2.0; +https://example.com/bot)"
RATE_DELAY = 1.0   # s entre requisições
GET_TIMEOUT = 30   # s

# Filtros mínimos de conteúdo
MIN_WORDS = 80
MIN_SCORE = 1

# Palavras-chave (PT/EN) — filtro leve
TERMS = [
    r"transforma[çc][aã]o digital",
    r"governo digital",
    r"setor p[úu]blico",
    r"servi[çc]os digitais",
    r"digital transformation",
    r"digital government",
    r"public sector",
    r"digital services",
    r"digital strategy",
]
TERMS_RX = [re.compile(t, re.IGNORECASE) for t in TERMS]

# ----------------------------
# Utilidades
# ----------------------------
def soft_score(text: str) -> int:
    return sum(1 for rx in TERMS_RX if rx.search(text or ""))

def word_count(text: str) -> int:
    return len(re.findall(r"\w+", text or ""))

def make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT})
    retries = Retry(
        total=3,
        backoff_factor=0.7,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"]),
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
            url TEXT UNIQUE,           -- URL do PDF
            lang TEXT,                 -- idioma da seed (pt/en)
            detected_lang TEXT,        -- idioma detectado no texto
            title TEXT,                -- título do PDF (metadata)
            num_pages INTEGER,         -- nº de páginas
            word_count INTEGER,        -- nº de palavras extraídas
            soft_score INTEGER,        -- score de termos-chave
            content TEXT,              -- texto extraído
            content_type TEXT,         -- "pdf"
            fetched_at TEXT            -- ISO-8601
        )
        """
    )
    # migrações idempotentes
    cur.execute("PRAGMA table_info(documents)")
    existing = {row[1] for row in cur.fetchall()}
    for col, ddl in [
        ("detected_lang", "TEXT"),
        ("title", "TEXT"),
        ("num_pages", "INTEGER"),
        ("word_count", "INTEGER"),
        ("soft_score", "INTEGER"),
    ]:
        if col not in existing:
            cur.execute(f"ALTER TABLE documents ADD COLUMN {col} {ddl}")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_documents_url ON documents(url)")
    conn.commit()

def init_db(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    ensure_table(conn)
    return conn

def load_seeds_file(path: Path, lang: str) -> list[tuple[str, str]]:
    items: list[tuple[str, str]] = []
    if not path.exists():
        return items
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            url = raw.strip()
            if url and not url.startswith("#"):
                items.append((url, lang))
    return items

def http_get_bytes(session: requests.Session, url: str) -> bytes:
    # sem limite de tamanho: baixa tudo em memória
    r = session.get(url, timeout=GET_TIMEOUT, stream=True, allow_redirects=True)
    r.raise_for_status()
    chunks: list[bytes] = []
    for chunk in r.iter_content(chunk_size=128 * 1024):
        if chunk:
            chunks.append(chunk)
    return b"".join(chunks)

def extract_pdf(raw: bytes) -> tuple[str, int, str]:
    """Retorna (texto_extraido, numero_de_paginas, titulo_pdf)."""
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

# heurística simples PT/EN sem libs externas
PT_STOP = {"de","da","do","dos","das","e","em","para","com","uma","um","que","por","não","como","ao","às","aos"}
EN_STOP = {"the","of","and","to","in","for","with","on","as","by","an","a","that","is","are","from"}

def detect_lang_simple(text: str) -> str:
    t = (text or "").lower()
    if not t:
        return ""
    score_pt = sum(1 for ch in t if ch in "áâãàéêíóôõúç") + sum(t.count(f" {w} ") for w in PT_STOP)
    score_en = sum(t.count(f" {w} ") for w in EN_STOP)
    if score_pt >= score_en + 1:
        return "pt"
    if score_en >= score_pt + 1:
        return "en"
    return "pt" if re.search(r"[ãõç]", t) else "en"

def save_document(
    conn: sqlite3.Connection,
    url: str,
    seed_lang: str,
    detected_lang: str,
    title: str,
    num_pages: int,
    wc: int,
    sscore: int,
    content: str,
) -> None:
    cur = conn.cursor()
    cur.execute(
        """
        INSERT OR REPLACE INTO documents
        (url, lang, detected_lang, title, num_pages, word_count, soft_score,
         content, content_type, fetched_at)
        VALUES (?,?,?,?,?,?,?,?,?,?)
        """,
        (
            url,
            seed_lang,
            detected_lang,
            title or None,
            num_pages,
            wc,
            sscore,
            content,
            "pdf",
            datetime.now(timezone.utc).isoformat(),
        ),
    )
    conn.commit()

# ----------------------------
# Núcleo (PDF-only)
# ----------------------------
def process_pdf(session: requests.Session, conn: sqlite3.Connection, url: str, seed_lang: str) -> None:
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
        print(f"[SKIP] PDF filtrado por conteúdo: {url}")
        return

    detected = detect_lang_simple(text)

    save_document(
        conn=conn,
        url=url,
        seed_lang=seed_lang,
        detected_lang=detected,
        title=title,
        num_pages=num_pages,
        wc=wc,
        sscore=sscore,
        content=text,
    )
    print(f"[OK] PDF salvo [{seed_lang}→{detected} | pgs={num_pages} | wc={wc} | score={sscore}]: {url}")

def crawl_pdf_only(seeds: list[tuple[str, str]]) -> None:
    conn = init_db(DB_PATH)
    session = make_session()

    # de-dup por URL mantendo o 1º idioma de seed
    unique: dict[str, str] = {}
    for url, lang in seeds:
        if url not in unique:
            unique[url] = lang

    print(f"Iniciando (PDF-only). Seeds únicas: {len(unique)}")
    for url, seed_lang in unique.items():
        try:
            process_pdf(session, conn, url, seed_lang)
        except Exception as e:
            print(f"[ERRO] Falha ao processar: {url} -> {e}")
        time.sleep(RATE_DELAY)

    conn.close()
    print(f"Concluído. Processados: {len(unique)} | DB: {DB_PATH}")

# ----------------------------
# Execução direta
# ----------------------------
def main() -> None:
    seeds_pt = load_seeds_file(SEEDS_FILE_PT, "pt")
    seeds_en = load_seeds_file(SEEDS_FILE_EN, "en")
    seeds = seeds_pt + seeds_en
    if not seeds:
        print("Nenhuma seed encontrada. Gere primeiro com: python seeds_finder.py")
        return
    crawl_pdf_only(seeds)

if __name__ == "__main__":
    main()
