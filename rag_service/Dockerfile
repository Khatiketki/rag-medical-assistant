FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    HF_HOME=/models PIP_NO_CACHE_DIR=1

WORKDIR /app/rag_service
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
COPY rag_service/requirements.txt .
RUN pip install -r requirements.txt

# Bake the embedding model into the image so containers start without network access.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5')"

COPY data /app/data
COPY rag_service /app/rag_service

RUN useradd --create-home appuser && mkdir -p /app/storage && chown -R appuser /app/storage
USER appuser

EXPOSE 8001
HEALTHCHECK --interval=15s --timeout=3s --start-period=60s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8001/healthz')"
CMD ["sh", "-c", "python -m app.ingest && uvicorn app.server:app --host 0.0.0.0 --port 8001"]
