#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script para orquestrar a execução do projeto de TCC (Chatbot).
Verifica artefatos existentes para evitar reprocessamento desnecessário.
"""

import os
import subprocess
import sys
import time

# --- Configuração dos Caminhos ---
# Todos os caminhos são relativos à raiz do projeto (onde este script deve estar)

# Arquivos de Requisitos
CHATBOT_REQ = "chatbot/requirements.txt"
API_REQ = "ingestion_api/requirements.txt"

# Caminhos dos Artefatos
SEEDS_FILE = "chatbot/extraction/seeds/seeds_pt.txt"
RAW_DB = "chatbot/extraction/data/knowledge_base.db"
PROCESSED_DB = "chatbot/data/knowledge_base_processed.db"
EMBEDDINGS_FILE = "chatbot/data/artifacts/sbert_embeddings.npy"
MAPPING_FILE = "chatbot/data/artifacts/sbert_mapping.parquet"
INDEX_FILE = "chatbot/data/artifacts/faiss.index"

# Comandos (módulos Python)
# Corrigidos para rodar da raiz do projeto
CMD_SEEDS = f"{sys.executable} -m chatbot.extraction.seeds.seeds_finder"
CMD_SCRAPING = f"{sys.executable} -m chatbot.extraction.scraping"
CMD_PROCESSOR = f"{sys.executable} -m chatbot.clean.processor"
CMD_EMBEDDER = f"{sys.executable} -m chatbot.ml.embedder"
CMD_BUILD_INDEX = f"{sys.executable} -m chatbot.ml.build_index"
CMD_DOCKER = "docker-compose up --build -d"
CMD_STREAMLIT = f"{sys.executable} -m streamlit run chatbot/ui/app.py"


# --- Funções Auxiliares ---

def print_header(message):
    """Imprime um cabeçalho formatado para separar as etapas."""
    print("\n" + "="*80)
    print(f"=== {message.upper()}")
    print("="*80)

def run_command(command, error_message="Ocorreu um erro na execução"):
    """
    Roda um comando no shell, imprime o output e para o script em caso de erro.
    """
    print(f"[INFO] Executando: {command}")
    try:
        # Usamos shell=True para compatibilidade com comandos como 'docker-compose'
        # e 'pip install -r'
        subprocess.run(command, check=True, shell=True)
        print(f"[SUCCESS] Comando concluído com sucesso.")
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] {error_message}")
        print(f"[ERROR] Detalhes: {e}")
        sys.exit(1) # Sai do script se um passo crítico falhar
    except FileNotFoundError as e:
        print(f"\n[ERROR] Comando não encontrado. Verifique se o executável (ex: 'docker-compose', 'python') está no PATH do sistema.")
        print(f"[ERROR] Detalhes: {e}")
        sys.exit(1)

def file_exists_and_non_empty(file_path):
    """Verifica se um arquivo existe e tem tamanho > 0."""
    return os.path.exists(file_path) and os.path.getsize(file_path) > 0

def db_exists_and_has_data(db_path, min_size_kb=8):
    """
    Verifica se um arquivo de banco de dados existe e tem um tamanho mínimo,
    sugerindo que contém dados (não apenas o schema).
    """
    if not os.path.exists(db_path):
        return False
    # Um DB SQLite vazio (só com schema) é bem pequeno.
    # Se tiver dados, rapidamente passa de 8KB.
    return os.path.getsize(db_path) > (min_size_kb * 1024)

# --- Lógica Principal ---

def main():
    print_header("Iniciando Script de Setup e Execução do Chatbot")
    
    # Passo 01: Instalar Dependências
    print_header("Passo 01: Instalando dependências (requirements.txt)")
    run_command(f"pip install -r {CHATBOT_REQ}", "Erro ao instalar dependências do chatbot.")
    run_command(f"pip install -r {API_REQ}", "Erro ao instalar dependências da API.")
    
    # Passo 02: Seeds Finder
    print_header("Passo 02: Verificando Seeds (extraction.seeds.seeds_finder)")
    if not file_exists_and_non_empty(SEEDS_FILE):
        print(f"[INFO] Arquivo '{SEEDS_FILE}' não encontrado ou vazio. Executando...")
        run_command(CMD_SEEDS, "Erro ao rodar o 'seeds_finder'.")
    else:
        print(f"[SKIP] Arquivo '{SEEDS_FILE}' já existe. Pulando etapa.")
        
    # Passo 03: Scraping
    print_header("Passo 03: Verificando DB Bruto (extraction.scraping)")
    if not db_exists_and_has_data(RAW_DB):
        print(f"[INFO] Banco '{RAW_DB}' não encontrado ou parece vazio. Executando...")
        run_command(CMD_SCRAPING, "Erro ao rodar o 'scraping'.")
    else:
        print(f"[SKIP] Banco '{RAW_DB}' já existe e contém dados. Pulando etapa.")

    # Passo 04: Processor
    print_header("Passo 04: Verificando DB Processado (clean.processor)")
    if not db_exists_and_has_data(PROCESSED_DB):
        print(f"[INFO] Banco '{PROCESSED_DB}' não encontrado ou parece vazio. Executando...")
        run_command(CMD_PROCESSOR, "Erro ao rodar o 'processor'.")
    else:
        print(f"[SKIP] Banco '{PROCESSED_DB}' já existe e contém dados. Pulando etapa.")

    # Passo 05: Embedder
    print_header("Passo 05: Verificando Arquivos de Embedding (ml.embedder)")
    if not (os.path.exists(EMBEDDINGS_FILE) and os.path.exists(MAPPING_FILE)):
        print(f"[INFO] Arquivos '{EMBEDDINGS_FILE}' ou '{MAPPING_FILE}' não encontrados. Executando...")
        run_command(CMD_EMBEDDER, "Erro ao rodar o 'embedder'.")
    else:
        print(f"[SKIP] Arquivos de embedding já existem. Pulando etapa.")

    # Passo 06: Build Index
    print_header("Passo 06: Verificando Índice FAISS (ml.build_index)")
    if not os.path.exists(INDEX_FILE):
        print(f"[INFO] Índice '{INDEX_FILE}' não encontrado. Executando...")
        run_command(CMD_BUILD_INDEX, "Erro ao rodar o 'build_index'.")
    else:
        print(f"[SKIP] Índice '{INDEX_FILE}' já existe. Pulando etapa.")
        
    # Passo 07: Docker (Ingestion API)
    print_header("Passo 07: Subindo Ingestion API (Docker Compose)")
    print("[INFO] Iniciando 'docker-compose up --build -d'. Isso pode levar um tempo...")
    run_command(CMD_DOCKER, "Erro ao iniciar o docker-compose. Verifique se o Docker está em execução.")
    print("[INFO] API de ingestão deve estar subindo em background.")
    time.sleep(5) # Pequena pausa para o docker "assentar"
    
    # Passo 08: Streamlit (UI)
    print_header("Passo 08: Iniciando Interface (Streamlit)")
    print("[INFO] Iniciando Streamlit. O terminal ficará preso nesta execução.")
    print("[INFO] Acesse o app no seu navegador (o endereço deve aparecer logo abaixo).")
    print("[INFO] Para parar o projeto, feche este terminal (Ctrl+C) e rode 'docker-compose down'.")
    
    # Este é o último comando e vai "travar" o terminal, que é o esperado.
    run_command(CMD_STREAMLIT, "Erro ao iniciar o Streamlit.")

if __name__ == "__main__":
    main()