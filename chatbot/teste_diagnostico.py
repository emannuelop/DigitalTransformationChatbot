#!/usr/bin/env python3
"""
Script de diagnóstico para verificar o estado do sistema RAG.
Execute este script para verificar se os PDFs do usuário estão sendo indexados corretamente.

Uso:
    python teste_diagnostico.py
"""

import sys
from pathlib import Path

# Adiciona o diretório do projeto ao path
# Ajuste conforme necessário para o seu ambiente
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

def test_database():
    """Verifica os bancos de dados."""
    print("=" * 80)
    print("TESTE 1: Verificando Bancos de Dados")
    print("=" * 80)
    
    import sqlite3
    
    # Banco bruto (extraction)
    try:
        db_path = PROJECT_ROOT / "extraction" / "data" / "knowledge_base.db"
        if not db_path.exists():
            print(f"❌ Banco bruto não encontrado: {db_path}")
        else:
            conn = sqlite3.connect(db_path)
            
            # Documentos globais
            cursor = conn.execute("SELECT COUNT(*) FROM documents WHERE user_id IS NULL")
            global_docs = cursor.fetchone()[0]
            print(f"✓ Documentos globais (user_id=NULL): {global_docs}")
            
            # Documentos de usuários
            cursor = conn.execute("SELECT COUNT(*) FROM documents WHERE user_id IS NOT NULL")
            user_docs = cursor.fetchone()[0]
            print(f"✓ Documentos de usuários (user_id!=NULL): {user_docs}")
            
            # Detalhes dos documentos de usuários
            if user_docs > 0:
                cursor = conn.execute("""
                    SELECT d.id, d.url, d.user_id, us.display_name 
                    FROM documents d
                    LEFT JOIN user_sources us ON us.doc_id = d.id
                    WHERE d.user_id IS NOT NULL
                    LIMIT 10
                """)
                print("\n  Documentos de usuários:")
                for row in cursor.fetchall():
                    print(f"    - ID: {row[0]}, URL: {row[1]}, User: {row[2]}, Nome: {row[3]}")
            
            conn.close()
    except Exception as e:
        print(f"❌ Erro ao verificar banco bruto: {e}")
    
    # Banco processado
    try:
        db_path = PROJECT_ROOT / "data" / "knowledge_base_processed.db"
        if not db_path.exists():
            print(f"\n❌ Banco processado não encontrado: {db_path}")
        else:
            conn = sqlite3.connect(db_path)
            
            # Chunks globais
            cursor = conn.execute("SELECT COUNT(*) FROM chunks WHERE user_id IS NULL")
            global_chunks = cursor.fetchone()[0]
            print(f"\n✓ Chunks globais (user_id=NULL): {global_chunks}")
            
            # Chunks de usuários
            cursor = conn.execute("SELECT COUNT(*) FROM chunks WHERE user_id IS NOT NULL")
            user_chunks = cursor.fetchone()[0]
            print(f"✓ Chunks de usuários (user_id!=NULL): {user_chunks}")
            
            # Detalhes dos chunks de usuários
            if user_chunks > 0:
                cursor = conn.execute("""
                    SELECT user_id, COUNT(*) as count
                    FROM chunks
                    WHERE user_id IS NOT NULL
                    GROUP BY user_id
                """)
                print("\n  Chunks por usuário:")
                for row in cursor.fetchall():
                    print(f"    - User ID {row[0]}: {row[1]} chunks")
            
            conn.close()
    except Exception as e:
        print(f"❌ Erro ao verificar banco processado: {e}")
    
    print()

def test_faiss_index():
    """Verifica o índice FAISS."""
    print("=" * 80)
    print("TESTE 2: Verificando Índice FAISS")
    print("=" * 80)
    
    try:
        import faiss
        import numpy as np
        
        index_path = PROJECT_ROOT / "data" / "artifacts" / "faiss.index"
        if not index_path.exists():
            print(f"❌ Índice FAISS não encontrado: {index_path}")
            return
        
        index = faiss.read_index(str(index_path))
        print(f"✓ Índice FAISS carregado")
        print(f"  - Total de vetores: {index.ntotal}")
        print(f"  - Dimensão: {index.d}")
        
        # Teste de busca simples
        if index.ntotal > 0:
            query = np.random.rand(1, index.d).astype('float32')
            D, I = index.search(query, min(5, index.ntotal))
            print(f"  - Teste de busca: OK (encontrou {len(I[0])} resultados)")
        
    except Exception as e:
        print(f"❌ Erro ao verificar índice FAISS: {e}")
    
    print()

def test_mapping():
    """Verifica o arquivo de mapping."""
    print("=" * 80)
    print("TESTE 3: Verificando Mapping Parquet")
    print("=" * 80)
    
    try:
        import pandas as pd
        
        mapping_path = PROJECT_ROOT / "data" / "artifacts" / "sbert_mapping.parquet"
        if not mapping_path.exists():
            print(f"❌ Mapping não encontrado: {mapping_path}")
            return
        
        mapping = pd.read_parquet(mapping_path)
        print(f"✓ Mapping carregado")
        print(f"  - Total de registros: {len(mapping)}")
        print(f"  - Colunas: {list(mapping.columns)}")
        
        # Verifica se tem coluna user_id
        if "user_id" in mapping.columns:
            global_count = mapping["user_id"].isna().sum()
            user_count = mapping["user_id"].notna().sum()
            print(f"  - Registros globais (user_id=NaN): {global_count}")
            print(f"  - Registros de usuários (user_id!=NaN): {user_count}")
            
            if user_count > 0:
                print("\n  Registros por usuário:")
                user_counts = mapping[mapping["user_id"].notna()].groupby("user_id").size()
                for user_id, count in user_counts.items():
                    print(f"    - User ID {user_id}: {count} registros")
                
                # Mostra alguns exemplos
                print("\n  Exemplos de registros de usuários:")
                user_samples = mapping[mapping["user_id"].notna()].head(3)
                for idx, row in user_samples.iterrows():
                    print(f"    - Índice {idx}: user_id={row['user_id']}, url={row.get('url', 'N/A')[:50]}...")
        else:
            print("  ⚠️  Coluna 'user_id' não encontrada no mapping!")
        
    except Exception as e:
        print(f"❌ Erro ao verificar mapping: {e}")
    
    print()

def test_search():
    """Testa a função de busca."""
    print("=" * 80)
    print("TESTE 4: Testando Busca")
    print("=" * 80)
    
    try:
        # Importa as funções do RAG
        from ml.rag_pipeline import load_search, search
        
        index, mapping = load_search()
        print("✓ Índice e mapping carregados")
        
        # Teste 1: Busca global (sem user_id)
        query1 = "transformação digital"
        results1 = search(index, mapping, query1, user_id=None, k=5)
        print(f"\n  Busca global: '{query1}'")
        print(f"    - Resultados: {len(results1)}")
        if len(results1) > 0:
            print(f"    - Top score: {results1.iloc[0]['score_vec']:.4f}")
            print(f"    - Top URL: {results1.iloc[0]['url'][:60]}...")
        
        # Teste 2: Busca com user_id (se houver usuários)
        if "user_id" in mapping.columns and mapping["user_id"].notna().any():
            user_ids = mapping[mapping["user_id"].notna()]["user_id"].unique()
            test_user_id = int(user_ids[0])
            
            query2 = "tesouro direto"
            results2 = search(index, mapping, query2, user_id=test_user_id, k=5)
            print(f"\n  Busca com user_id={test_user_id}: '{query2}'")
            print(f"    - Resultados: {len(results2)}")
            if len(results2) > 0:
                print(f"    - Top score: {results2.iloc[0]['score_vec']:.4f}")
                print(f"    - Top URL: {results2.iloc[0]['url'][:60]}...")
                print(f"    - User IDs nos resultados: {results2['user_id'].unique()}")
        
    except Exception as e:
        print(f"❌ Erro ao testar busca: {e}")
        import traceback
        traceback.print_exc()
    
    print()

def test_rag_pipeline():
    """Testa o pipeline RAG completo."""
    print("=" * 80)
    print("TESTE 5: Testando Pipeline RAG Completo")
    print("=" * 80)
    
    try:
        from ml.rag_pipeline import answer_with_cfg
        
        # Teste 1: Pergunta sobre transformação digital (global)
        question1 = "O que é transformação digital?"
        print(f"\n  Pergunta global: '{question1}'")
        answer1, urls1, debug1 = answer_with_cfg(question1, user_id=None, k=5)
        print(f"    - Resposta: {answer1[:100]}...")
        print(f"    - URLs: {len(urls1)}")
        print(f"    - Debug: {debug1}")
        
        # Teste 2: Pergunta com user_id (se houver)
        import pandas as pd
        mapping_path = PROJECT_ROOT / "data" / "artifacts" / "sbert_mapping.parquet"
        mapping = pd.read_parquet(mapping_path)
        
        if "user_id" in mapping.columns and mapping["user_id"].notna().any():
            user_ids = mapping[mapping["user_id"].notna()]["user_id"].unique()
            test_user_id = int(user_ids[0])
            
            question2 = "Quais são os principais tópicos deste documento?"
            print(f"\n  Pergunta com user_id={test_user_id}: '{question2}'")
            answer2, urls2, debug2 = answer_with_cfg(question2, user_id=test_user_id, k=5)
            print(f"    - Resposta: {answer2[:100]}...")
            print(f"    - URLs: {len(urls2)}")
            print(f"    - Debug: {debug2}")
        
    except Exception as e:
        print(f"❌ Erro ao testar pipeline RAG: {e}")
        import traceback
        traceback.print_exc()
    
    print()

def main():
    """Executa todos os testes."""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "DIAGNÓSTICO DO SISTEMA RAG" + " " * 32 + "║")
    print("╚" + "=" * 78 + "╝")
    print()
    
    test_database()
    test_faiss_index()
    test_mapping()
    test_search()
    test_rag_pipeline()
    
    print("=" * 80)
    print("DIAGNÓSTICO CONCLUÍDO")
    print("=" * 80)
    print()

if __name__ == "__main__":
    main()
