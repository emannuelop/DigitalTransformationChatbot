import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent / "data" / "knowledge_base.db"

def main():
    if not Path(DB_PATH).exists():
        print(f"Banco {DB_PATH} não encontrado. Rode primeiro: python crawler.py")
        return

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # Quantidade total
    cur.execute("SELECT COUNT(*) FROM documents")
    total = cur.fetchone()[0]

    # Por tipo
    cur.execute("SELECT content_type, COUNT(*) FROM documents GROUP BY content_type")
    counts = cur.fetchall()

    print(f"Banco: {DB_PATH}")
    print(f"Total de documentos: {total}")
    for ctype, count in counts:
        print(f"  - {ctype}: {count}")

    conn.close()

if __name__ == "__main__":
    main()
