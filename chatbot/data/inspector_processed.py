# inspector_processed.py
import argparse
import sqlite3
from pathlib import Path
from typing import Optional

import pandas as pd

def connect_db(db_path: Path) -> sqlite3.Connection:
    if not db_path.exists():
        raise FileNotFoundError(f"Banco processado não encontrado: {db_path}")
    return sqlite3.connect(str(db_path))

def q(conn: sqlite3.Connection, sql: str, params: Optional[tuple] = None) -> pd.DataFrame:
    return pd.read_sql_query(sql, conn, params=params or ())

def print_header(title: str):
    print("\n" + "=" * 88)
    print(title)
    print("=" * 88)

def main():
    # paths
    DATA_DIR = Path(__file__).resolve().parent
    DB_PATH = DATA_DIR / "knowledge_base_processed.db"
    ARTIFACTS = DATA_DIR / "artifacts"

    ap = argparse.ArgumentParser(description="Inspector do banco processado (Etapa 02)")
    ap.add_argument("--limit", type=int, default=3, help="Qtde de exemplos para imprimir.")
    ap.add_argument("--no-examples", action="store_true", help="Não mostrar exemplos de documentos.")
    ap.add_argument("--export-csv", action="store_true", help="Exporta tabelas para CSVs em data/exports.")
    args = ap.parse_args()

    conn = connect_db(DB_PATH)

    # 1) Visão geral
    print_header("1) Visão geral")
    total_docs = q(conn, "SELECT COUNT(*) AS n FROM processed_documents")["n"].iloc[0]
    total_dups = q(conn, "SELECT COUNT(*) AS n FROM duplicates_map")["n"].iloc[0]
    print(f"Banco: {DB_PATH}")
    print(f"Docs processados: {total_docs}")
    print(f"Entradas em duplicates_map: {total_dups}")

    # 2) Distribuições
    print_header("2) Distribuições (idioma / categoria)")
    by_lang = q(conn, """
        SELECT lang, COUNT(*) AS n, ROUND(100.0*COUNT(*)/(SELECT COUNT(*) FROM processed_documents),2) AS pct
        FROM processed_documents GROUP BY lang ORDER BY n DESC
    """)
    by_cat = q(conn, """
        SELECT category, COUNT(*) AS n, ROUND(100.0*COUNT(*)/(SELECT COUNT(*) FROM processed_documents),2) AS pct
        FROM processed_documents GROUP BY category ORDER BY n DESC
    """)
    print("Por idioma:")
    print(by_lang.to_string(index=False))
    print("\nPor categoria:")
    print(by_cat.to_string(index=False))

    # 3) Sanity checks (hashes únicos / órfãos em duplicates_map)
    print_header("3) Sanity checks")
    unique_hashes = q(conn, "SELECT COUNT(DISTINCT text_hash) AS uniq_hash FROM processed_documents")["uniq_hash"].iloc[0]
    print(f"Hashes únicos em processed_documents: {unique_hashes}")

    orfaos = q(conn, """
        SELECT COUNT(*) AS n
        FROM duplicates_map dm
        LEFT JOIN processed_documents pd
        ON dm.kept_original_id = pd.original_id
        WHERE pd.original_id IS NULL
    """)["n"].iloc[0]
    print(f"Duplicatas mapeadas para 'mantidos' inexistentes: {orfaos}")

    # 4) Exemplos
    if not args.no_examples and total_docs > 0:
        print_header(f"4) Exemplos (mostrando {args.limit})")
        samples = q(conn, """
            SELECT original_id, url, lang, category, content_len,
                   substr(content_clean, 1, 300) AS preview
            FROM processed_documents
            ORDER BY content_len DESC
            LIMIT ?
        """, (args.limit,))
        for _, r in samples.iterrows():
            print("-" * 88)
            print(f"original_id: {r['original_id']} | lang: {r['lang']} | cat: {r['category']} | len: {r['content_len']}")
            print(f"url: {r['url']}")
            print(f"preview: {r['preview']}...")
        print("-" * 88)

    # 5) Artifacts
    print_header("5) Artifacts (TF-IDF / SBERT)")
    tfidf_vec = ARTIFACTS / "tfidf_vectorizer.pkl"
    tfidf_mat = ARTIFACTS / "tfidf_matrix.npz"
    sbert_npy = ARTIFACTS / "sbert_embeddings.npy"
    sbert_model_txt = ARTIFACTS / "sbert_model.txt"

    def ok(p: Path) -> str:
        return f"OK  {p}" if p.exists() else f"FALTA  {p}"

    print(ok(tfidf_vec))
    print(ok(tfidf_mat))
    print(ok(sbert_npy))
    print(ok(sbert_model_txt))

    # tenta reportar shapes
    try:
        from scipy import sparse
        import joblib, numpy as np
        if tfidf_mat.exists():
            X = sparse.load_npz(tfidf_mat)
            print(f"TF-IDF shape: {X.shape}")
        if sbert_npy.exists():
            embs = np.load(sbert_npy)
            print(f"SBERT shape: {embs.shape}")
        if tfidf_vec.exists():
            vec = joblib.load(tfidf_vec)
            print(f"Vocabulário TF-IDF: {len(getattr(vec, 'vocabulary_', {}))} termos")
        if sbert_model_txt.exists():
            print("SBERT modelo:", sbert_model_txt.read_text(encoding="utf-8").strip())
    except Exception as e:
        print(f"Aviso: não foi possível inspecionar artifacts ({e})")

    # 6) Export opcional
    if args.export_csv and total_docs > 0:
        export_dir = DATA_DIR / "exports"
        export_dir.mkdir(parents=True, exist_ok=True)
        q(conn, "SELECT * FROM processed_documents").to_csv(export_dir / "processed_documents.csv", index=False, encoding="utf-8")
        q(conn, "SELECT * FROM duplicates_map").to_csv(export_dir / "duplicates_map.csv", index=False, encoding="utf-8")
        print_header("Exportação")
        print(f"Tabelas exportadas para: {export_dir}")

    conn.close()
    print("\nConcluído.")

if __name__ == "__main__":
    main()
