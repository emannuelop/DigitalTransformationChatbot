# 🤖 Digital Transformation Chatbot

Um chatbot para apoiar jornadas de transformação digital no setor público utilizando
**RAG (Retrieval-Augmented Generation)**, extração de dados, vetorização
com **FAISS** e uma interface web simples.

Este projeto integra:

- **LM Studio** como provedor local de LLM\
- **Modelo `ibm/granite-4-h-tiny`**\
- **Streamlit** para UI\
- **Docker** para a API de ingestão\
- **SQLite** para persistência

## ✅ Pré-requisitos

Instale e/ou tenha disponível:

- **Python 3.10+**
- **Docker** e **Docker Compose**
- **LM Studio**

⚠️ O **LM Studio deve estar inicializado** e o **modelo carregado**
antes de rodar o projeto.

## 🧠 Passos no LM Studio

1. Abra o **LM Studio**
2. Baixe o modelo `ibm/granite-4-h-tiny`
3. Vá em **Server → Start Local Server**
4. Mantenha o servidor rodando durante o uso

## 🛠️ Instalação e Execução

### 1 Clonar o repositório

```bash
git clone https://github.com/SEU_USUARIO/DigitalTransformationChatbot.git
cd DigitalTransformationChatbot
```

### 1 Executar o projeto (raiz)

```bash
python run_project.py
```

### 3 Acessar a interface

```bash
http://localhost:8501
```

## 👨‍💻 Autores

<!-- markdownlint-capture -->
<!-- markdownlint-disable MD033 MD045 MD047 -->

| <img src="imagens/emannuel.png" alt="Foto de Emannuel Oliveira" width="115"><br><sub>Emannuel Oliveira</sub> |
| :---: |