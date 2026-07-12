FROM python:3.10-slim

WORKDIR /app

COPY _development/amr-gff-nn/deploy/requirements.txt /app/deploy/requirements.txt
RUN pip install --no-cache-dir -r /app/deploy/requirements.txt

COPY _development/amr-gff-nn/src /app/src
COPY _development/amr-gff-nn/deploy /app/deploy
COPY _development/amr-gff-nn/artifacts/gff-v3 /app/artifacts/gff-v3

RUN useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"

CMD ["uvicorn", "deploy.service.app:app", "--host", "0.0.0.0", "--port", "8000"]