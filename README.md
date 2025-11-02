# DigitalTransformationChatbot

Bem-vindo ao **DigitalTransformationChatbot**, um projeto de chatbot focado em Transformação Digital no Setor Público. Este chatbot utiliza uma arquitetura RAG (Retrieval-Augmented Generation) para fornecer respostas informadas e contextuais, alimentado por uma base de conhecimento construída a partir de scraping web e ingestão de documentos.

O sistema é composto por uma interface de usuário web (construída com Streamlit), um pipeline de machine learning para processamento e recuperação de dados, e uma API de ingestão de dados.

## 🏛️ Arquitetura

O projeto é dividido nos seguintes componentes principais:

* **`chatbot/ui`**: A interface do usuário web (Streamlit) onde os usuários podem interagir com o chatbot, gerenciar perfis e fazer upload de novos documentos.
* **`chatbot/ml`**: O núcleo do pipeline RAG. Contém a lógica para:
* `embedder.py`: Gerar embeddings de texto.
* `build_index.py`: Construir e salvar um índice FAISS para busca rápida de vetores.
* `llm_backends.py`: Conectar-se a modelos de linguagem (como o `ibm/granite-4-h-tiny` via LM Studio).
* `rag_pipeline.py`: Orquestrar a lógica de recuperação de contexto e geração de resposta.
* **`chatbot/extraction`**: Scripts responsáveis pela coleta de dados (`scraping.py`) e ingestão de dados do usuário (`user_ingest.py`).
* **`chatbot/clean`**: Módulo para processamento e limpeza dos dados extraídos.
* **`ingestion_api`**: Uma API (provavelmente FastAPI/Flask) containerizada com Docker, responsável por lidar com os processos de ingestão de dados em segundo plano.
* **`run_project.py`**: Script principal na raiz do projeto para orquestrar e iniciar todos os serviços.

## 🚀 Pré-requisitos

Antes de iniciar, garanta que você tenha os seguintes softwares instalados e configurados em sua máquina:

1. **[Python](https://www.python.org/downloads/)**: Necessário para rodar os scripts da aplicação e a interface web.
2. **[Docker](https://www.docker.com/products/docker-desktop/)**: Necessário para rodar a `ingestion_api` e outros serviços containerizados definidos no `docker-compose.yml`.
3. **[LM Studio](https://lmstudio.ai/)**: Necessário para baixar e servir o modelo de linguagem localmente.

## ⚙️ Instalação e Configuração

Siga este passo a passo para configurar o ambiente e rodar o projeto.

### 1. Instale as Dependências Python

Navegue até a pasta `chatbot` e instale todas as bibliotecas necessárias:

```bash
cd chatbot
pip install -r requirements.txt
cd .. 
# Volte para a raiz do projeto
