# Imagen de servicio de mibici-demand: API FastAPI con el modelo exportado en models/.
# Misma versión de Python que el .venv que entrenó el modelo (3.12.2).
# Base multiplataforma: compatible con
#   docker buildx build --platform linux/amd64,linux/arm64 .
FROM python:3.12.2-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app

WORKDIR /app

# Primero las dependencias, para reutilizar la capa si solo cambia el código
COPY requirements-serving.txt .
RUN pip install -r requirements-serving.txt

# Código, parámetros y modelo; quedan de root (solo lectura para el servicio)
COPY params.yaml ./
COPY mibici/ mibici/
COPY models/ models/

# mibici.config crea data/* al importarse: se crean aquí porque el usuario del
# servicio no puede escribir en /app
RUN mkdir -p data/raw data/interim data/processed data/external \
    && useradd --create-home --uid 1000 app

USER app

EXPOSE 8000

# La imagen slim no trae curl: el chequeo se hace con Python
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=3)"

CMD ["uvicorn", "mibici.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
