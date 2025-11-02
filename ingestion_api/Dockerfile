# ingestion_api/Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Instala dependências da API
COPY ./ingestion_api/requirements.txt /app/ingestion_api/requirements.txt
RUN pip install --no-cache-dir -r /app/ingestion_api/requirements.txt

# Instala dependências do SEU CÓDIGO (o chatbot)
COPY ./chatbot/requirements.txt /app/chatbot/requirements.txt
RUN pip install --no-cache-dir -r /app/chatbot/requirements.txt

# Copia todo o seu código
COPY ./chatbot /app/chatbot
COPY ./ingestion_api /app/ingestion_api

EXPOSE 8000
CMD ["uvicorn", "ingestion_api.main:app", "--host", "0.0.0.0", "--port", "8000"]