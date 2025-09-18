import random
import time

from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from ddgs import DDGS

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
SEEDS_DIR = Path(__file__).parent
SEEDS_FILE_PT = SEEDS_DIR / "seeds_pt.txt"
SEEDS_FILE_EN = SEEDS_DIR / "seeds_en.txt"

# ---------------------------------------------------------------------
# Consultas padrão (PDF-only)
# ---------------------------------------------------------------------
QUERIES_PT = [
    '"transformação digital" filetype:pdf',
    '"estratégia de transformação digital" filetype:pdf',
    '"transformação digital setor público" filetype:pdf',
    '"inovação tecnológica" transformação digital filetype:pdf',
]

QUERIES_EN = [
    '"digital transformation" filetype:pdf',
    '"digital transformation strategy" filetype:pdf',
    '"digital transformation public sector" filetype:pdf',
    '"case study" "digital transformation" filetype:pdf',
]

# ---------------------------------------------------------------------
# Blocklist simples
# ---------------------------------------------------------------------
BLOCKED_HOST_SUFFIXES = {
    "pinterest.com", "br.pinterest.com",
    "facebook.com", "m.facebook.com", "web.facebook.com",
    "instagram.com", "x.com", "twitter.com", "t.co",
    "tiktok.com", "whatsapp.com",
    "linktr.ee", "bit.ly", "goo.gl", "tinyurl.com", "lnkd.in",
    "reddit.com", "www.reddit.com", "discord.com",
}

# ---------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------
def normalize(url: str) -> str:
    if not url:
        return url
    parts = urlsplit(url.strip())
    parts = parts._replace(fragment="")

    # limpa tracking
    blocked_params = {"fbclid", "gclid"}
    qs = [(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True)
          if not k.lower().startswith("utm_") and k.lower() not in blocked_params]
    parts = parts._replace(query=urlencode(qs, doseq=True))

    # normaliza esquema/host
    parts = parts._replace(scheme=(parts.scheme or "").lower(),
                           netloc=(parts.netloc or "").lower())

    cleaned = urlunsplit(parts)
    base = f"{parts.scheme}://{parts.netloc}"
    if cleaned.endswith("/") and len(cleaned) > len(base) + 1:
        cleaned = cleaned[:-1]
    return cleaned

def load_existing(path: Path) -> set[str]:
    if not path.exists():
        return set()
    out: set[str] = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s and not s.startswith("#"):
                out.add(s)
    return out

def save_merged(path: Path, new_items: set[str]) -> None:
    header, old_urls = [], []
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip() or line.startswith("#"):
                    header.append(line)
                else:
                    old_urls.append(line.strip())

    merged, seen = [], set()
    for u in old_urls:
        if u not in seen:
            merged.append(u); seen.add(u)
    for u in sorted(new_items):
        if u not in seen:
            merged.append(u); seen.add(u)

    with open(path, "w", encoding="utf-8") as f:
        if header:
            f.writelines(header)
        for u in merged:
            f.write(u + "\n")

def is_pdf_url(u: str) -> bool:
    return u.lower().endswith(".pdf")

def allowed_host(u: str) -> bool:
    host = (urlsplit(u).netloc or "").lower()
    if not host:
        return False
    for suf in BLOCKED_HOST_SUFFIXES:
        if host == suf or host.endswith("." + suf):
            return False
    return True

def ddg_search(query: str, lang: str, limit: int = 15) -> list[str]:
    region = "br-pt" if lang == "pt" else "wt-wt"
    out: list[str] = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, region=region, safesearch="moderate", max_results=limit):
            href = r.get("href") or r.get("link") or r.get("url")
            if href:
                out.append(href)
    return out[:limit]

# ---------------------------------------------------------------------
# Execução direta
# ---------------------------------------------------------------------
def main() -> None:
    existing_pt = load_existing(SEEDS_FILE_PT)
    existing_en = load_existing(SEEDS_FILE_EN)
    new_pt: set[str] = set()
    new_en: set[str] = set()

    all_queries = [("pt", QUERIES_PT, existing_pt, new_pt, SEEDS_FILE_PT),
                   ("en", QUERIES_EN, existing_en, new_en, SEEDS_FILE_EN)]

    step, total = 0, sum(len(qs) for _, qs, *_ in all_queries)

    for lang, queries, existing, collected, out_path in all_queries:
        for q in queries:
            step += 1
            try:
                results = ddg_search(q, lang)
            except Exception as e:
                print(f"[WARN] Falha na busca '{q}' ({lang}): {e}")
                results = []
            added = 0
            for url in results:
                u = normalize(url)
                if not u or not is_pdf_url(u):
                    continue
                if not allowed_host(u):
                    continue
                if u in existing or u in collected:
                    continue
                collected.add(u)
                added += 1
            print(f"[OK] {step}/{total} '{q}' [{lang}] -> +{added} PDFs")
            time.sleep(1.0 + random.uniform(0, 0.4))
        if collected:
            save_merged(out_path, collected)

    print("\nResumo final:")
    print(f"  PT: +{len(new_pt)} PDFs novos -> {SEEDS_FILE_PT}")
    print(f"  EN: +{len(new_en)} PDFs novos -> {SEEDS_FILE_EN}")
    if not new_pt and not new_en:
        print("Nenhum PDF novo encontrado.")

if __name__ == "__main__":
    main()