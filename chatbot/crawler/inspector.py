import sqlite3

from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from urllib.parse import urlsplit

DB_PATH = Path(__file__).parent / "data" / "knowledge_base.db"

def as_dt(iso: str) -> str:
    """Formata ISO-8601 -> 'YYYY-mm-dd HH:MM' local (sem TZ)."""
    try:
        return datetime.fromisoformat(iso.replace("Z", "+00:00")).strftime("%Y-%m-%d %H:%M")
    except Exception:
        return iso or ""


def open_db(db_path: Path) -> sqlite3.Connection:
    if not db_path.exists():
        print(f"Banco {db_path} não encontrado. Rode primeiro: python crawler.py")
        raise SystemExit(0)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn

def fetchall(conn: sqlite3.Connection, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
    cur = conn.cursor()
    cur.execute(sql, params)
    return cur.fetchall()

def fetchone(conn: sqlite3.Connection, sql: str, params: tuple = ()) -> sqlite3.Row | None:
    cur = conn.cursor()
    cur.execute(sql, params)
    return cur.fetchone()

def print_header(title: str) -> None:
    print("\n" + title)
    print("-" * len(title))

def main() -> None:
    conn = open_db(DB_PATH)

    # -------- panorama geral --------
    total_row = fetchone(conn, "SELECT COUNT(*) AS n FROM documents")
    total = total_row["n"] if total_row else 0

    by_type = fetchall(conn, "SELECT content_type, COUNT(*) AS n FROM documents GROUP BY content_type ORDER BY n DESC")
    by_seed_lang = fetchall(conn, "SELECT COALESCE(lang,'') AS lang, COUNT(*) AS n FROM documents GROUP BY COALESCE(lang,'') ORDER BY n DESC")
    by_detected_lang = fetchall(conn, "SELECT COALESCE(detected_lang,'') AS dlang, COUNT(*) AS n FROM documents GROUP BY COALESCE(detected_lang,'') ORDER BY n DESC")

    print_header("Banco")
    print(f"Caminho: {DB_PATH}")
    print(f"Total de documentos: {total}")

    print_header("Por content_type")
    if by_type:
        for r in by_type:
            print(f"  - {r['content_type'] or '(vazio)'}: {r['n']}")
    else:
        print("  (sem registros)")

    print_header("Por idioma da seed (lang)")
    for r in by_seed_lang:
        lab = r["lang"] or "(vazio)"
        print(f"  - {lab}: {r['n']}")

    print_header("Por idioma detectado (detected_lang)")
    for r in by_detected_lang:
        lab = r["dlang"] or "(vazio)"
        print(f"  - {lab}: {r['n']}")

    # -------- cruzamento seed x detect --------
    cross = fetchall(
        conn,
        """
        SELECT COALESCE(lang,'') AS seed_lang,
               COALESCE(detected_lang,'') AS detected_lang,
               COUNT(*) AS n
        FROM documents
        GROUP BY COALESCE(lang,''), COALESCE(detected_lang,'')
        ORDER BY n DESC
        """
    )
    print_header("Cruzamento: seed lang x detected lang")
    if not cross:
        print("  (sem registros)")
    else:
        for r in cross:
            s = r["seed_lang"] or "(vazio)"
            d = r["detected_lang"] or "(vazio)"
            print(f"  - {s} → {d}: {r['n']}")

    # -------- estatísticas de tamanho --------
    stats = fetchall(conn, "SELECT word_count AS wc, num_pages AS p FROM documents WHERE word_count IS NOT NULL")
    wcs = [r["wc"] for r in stats if isinstance(r["wc"], int)]
    pages = [r["p"] for r in stats if isinstance(r["p"], int)]

    print_header("Estatísticas de tamanho")
    if wcs:
        avg_wc = sum(wcs) / len(wcs)
        p50_wc = sorted(wcs)[len(wcs) // 2]
        print(f"  Palavras — média: {avg_wc:.1f} | mediana: {p50_wc} | min: {min(wcs)} | max: {max(wcs)}")
    else:
        print("  Sem word_count registrado.")

    if pages:
        avg_pg = sum(pages) / len(pages)
        p50_pg = sorted(pages)[len(pages) // 2]
        print(f"  Páginas — média: {avg_pg:.2f} | mediana: {p50_pg} | min: {min(pages)} | max: {max(pages)}")
    else:
        print("  Sem num_pages registrado.")

    # -------- top domínios --------
    urls = fetchall(conn, "SELECT url FROM documents WHERE url IS NOT NULL")
    domain_counter = Counter()
    for r in urls:
        host = (urlsplit(r["url"]).netloc or "").lower()
        domain_counter[host] += 1

    print_header("Top domínios (TOP 10)")
    for host, n in domain_counter.most_common(10):
        print(f"  - {host or '(vazio)'}: {n}")

    # -------- últimos documentos --------
    last = fetchall(
        conn,
        """
        SELECT url, title, lang, detected_lang, num_pages, word_count, soft_score, fetched_at
        FROM documents
        ORDER BY datetime(fetched_at) DESC
        LIMIT 8
        """
    )
    print_header("Mais recentes (TOP 8)")
    if not last:
        print("  (sem registros)")
    else:
        for r in last:
            print(f"• {r['title'] or '(sem título)'}")
            print(f"    URL: {r['url']}")
            print(f"    Seed/Detect: {r['lang'] or '-'} → {r['detected_lang'] or '-'}")
            print(f"    Páginas/Palavras/Score: {r['num_pages'] or 0} / {r['word_count'] or 0} / {r['soft_score'] or 0}")
            print(f"    Quando: {as_dt(r['fetched_at'] or '')}")

    # -------- possíveis anomalias --------
    anomalies: dict[str, list[str]] = defaultdict(list)

    # conteúdo vazio
    empty = fetchall(conn, "SELECT url FROM documents WHERE content IS NULL OR TRIM(content) = ''")
    if empty:
        anomalies["Sem conteúdo extraído"].extend([r["url"] for r in empty])

    # word_count abaixo do mínimo esperado
    low_wc = fetchall(conn, "SELECT url, word_count FROM documents WHERE word_count < 80")
    if low_wc:
        for r in low_wc:
            anomalies["word_count < 80"].append(f"{r['url']} (wc={r['word_count']})")

    # inconsistência de idiomas seed vs detect
    mismatch = fetchall(
        conn,
        "SELECT url, lang, detected_lang FROM documents WHERE lang IS NOT NULL AND detected_lang IS NOT NULL AND lang <> detected_lang"
    )
    if mismatch:
        for r in mismatch:
            anomalies["Seed lang ≠ detected"].append(f"{r['url']} ({r['lang']}→{r['detected_lang']})")

    print_header("Possíveis anomalias")
    if not anomalies:
        print("  Nenhuma anomalia simples encontrada.")
    else:
        for kind, items in anomalies.items():
            print(f"  - {kind}: {len(items)}")
            for u in items[:10]:  # limita a listagem
                print(f"      • {u}")
            if len(items) > 10:
                print(f"      (+{len(items)-10} mais)")

    conn.close()

if __name__ == "__main__":
    main()