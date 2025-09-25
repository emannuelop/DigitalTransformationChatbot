import random
import time
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode
from ddgs import DDGS

SEEDS_DIR = Path(__file__).parent
SEEDS_FILE = SEEDS_DIR / "seeds_pt.txt"
SEEDS_DIR.mkdir(parents=True, exist_ok=True)

QUERIES_PT = [
    # Geral
    '"transformação digital" filetype:pdf',
    '"estratégia de transformação digital" filetype:pdf',
    '"indústria 4.0" transformação digital filetype:pdf',
    '"maturidade digital" filetype:pdf',
    '"mapa de jornada digital" filetype:pdf',
    '"roadmap de transformação digital" filetype:pdf',
    '"capacidade digital" organização filetype:pdf',
    '"case" "transformação digital" filetype:pdf',
    '"indicadores de transformação digital" filetype:pdf',
    '"pequenas e médias empresas" "transformação digital" filetype:pdf',

    # Setor público
    '"governo digital" filetype:pdf',
    '"estratégia de governo digital" filetype:pdf',
    '"administração pública" "transformação digital" filetype:pdf',
    '"setor público" "transformação digital" filetype:pdf',
    '"serviços públicos digitais" filetype:pdf',
    '"cidades inteligentes" "transformação digital" filetype:pdf',
    '"transformação digital" "gestão pública" filetype:pdf',
    '"transformação digital" "políticas públicas" filetype:pdf',
    '"governança digital" setor público filetype:pdf',
    '"interoperabilidade" "transformação digital" governo filetype:pdf',
    '"dados abertos" "transformação digital" filetype:pdf',
    '"gov.br" transformação digital filetype:pdf',

    # Áreas específicas
    '"saúde digital" "transformação digital" filetype:pdf',
    '"educação digital" "transformação digital" filetype:pdf',
    '"justiça digital" "transformação digital" filetype:pdf',
]

BLOCKED_HOST_SUFFIXES = {
    "pinterest.com", "br.pinterest.com",
    "facebook.com", "m.facebook.com", "web.facebook.com",
    "instagram.com", "x.com", "twitter.com", "t.co",
    "tiktok.com", "whatsapp.com",
    "linktr.ee", "bit.ly", "goo.gl", "tinyurl.com", "lnkd.in",
    "reddit.com", "www.reddit.com", "discord.com",
}

HEAD_CHECK_IF_NEEDED = False

# =============================================================================
# Utils
# =============================================================================
def normalize(url: str) -> str:
    if not url:
        return url
    parts = urlsplit(url.strip())
    parts = parts._replace(fragment="")

    blocked_params = {"fbclid", "gclid"}
    qs = [
        (k, v)
        for k, v in parse_qsl(parts.query, keep_blank_values=True)
        if not k.lower().startswith("utm_") and k.lower() not in blocked_params
    ]
    parts = parts._replace(query=urlencode(qs, doseq=True))
    parts = parts._replace(
        scheme=(parts.scheme or "").lower(),
        netloc=(parts.netloc or "").lower(),
    )

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
            merged.append(u)
            seen.add(u)
    for u in sorted(new_items):
        if u not in seen:
            merged.append(u)
            seen.add(u)

    with open(path, "w", encoding="utf-8") as f:
        if header:
            f.writelines(header)
        for u in merged:
            f.write(u + "\n")

def allowed_host(url: str) -> bool:
    host = (urlsplit(url).netloc or "").lower()
    if not host:
        return False
    return not any(host == suf or host.endswith("." + suf) for suf in BLOCKED_HOST_SUFFIXES)

def is_pdf_like(url: str) -> bool:
    return ".pdf" in (urlsplit(url).path or "").lower()

def confirm_pdf_head(url: str) -> bool:
    if not HEAD_CHECK_IF_NEEDED:
        return True
    try:
        import requests
        r = requests.head(url, allow_redirects=True, timeout=10)
        ct = (r.headers.get("Content-Type") or "").lower()
        return "application/pdf" in ct
    except Exception:
        return False

def ddg_search_pt(query: str, limit: int = 20) -> list[str]:
    out: list[str] = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, region="br-pt", safesearch="moderate", max_results=limit):
            href = r.get("href") or r.get("link") or r.get("url")
            if href:
                out.append(href)
    return out[:limit]

def main() -> None:
    existing = load_existing(SEEDS_FILE)
    collected: set[str] = set()

    total = len(QUERIES_PT)
    print(f"Iniciando busca PT-BR (PDFs). Consultas: {total}\nArquivo de saída: {SEEDS_FILE}\n")

    for i, q in enumerate(QUERIES_PT, start=1):
        try:
            results = ddg_search_pt(q)
        except Exception as e:
            print(f"[WARN] Falha na busca '{q}': {e}")
            results = []

        added = 0
        for raw in results:
            u = normalize(raw)
            if not u or not allowed_host(u):
                continue
            if not is_pdf_like(u):
                continue
            if not confirm_pdf_head(u):
                continue
            if u in existing or u in collected:
                continue

            collected.add(u)
            added += 1

        print(f"[OK] {i}/{total} '{q}' -> +{added} PDFs")
        time.sleep(1.0 + random.uniform(0, 0.4))  # educado com o buscador

    if collected:
        save_merged(SEEDS_FILE, collected)

    print("\nResumo final:")
    print(f"  PT: +{len(collected)} PDFs novos -> {SEEDS_FILE}")
    if not collected:
        print("Nenhum PDF novo encontrado.")

if __name__ == "__main__":
    main()