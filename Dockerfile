# Imagen para correr el pipeline batch (spark-submit). A diferencia de la API de P1, este
# job SÍ necesita Java (Spark 4.x) además de Python, así que instalamos un JRE 17.
FROM python:3.12-slim AS base

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1

# JRE 21 headless: Spark 4.x requiere Java 17/21 (no soporta 25). procps para diagnósticos.
RUN apt-get update \
    && apt-get install -y --no-install-recommends openjdk-21-jre-headless procps \
    && rm -rf /var/lib/apt/lists/*
ENV JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64

# uv desde la imagen oficial (sin instalar a nivel de sistema).
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# 1) Dependencias primero, para aprovechar la cache de capas.
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --no-dev --frozen

# 2) Configs y scripts del pipeline.
COPY configs ./configs
COPY scripts ./scripts

# mock por defecto: el contenedor corre sin descargar modelos. Cambia a hf con -e GENAI_BACKEND=hf.
ENV PATH="/app/.venv/bin:$PATH" \
    GENAI_BACKEND=mock

# Job batch (no servicio): corre un dominio y termina. El config es sobreescribible vía CMD.
ENTRYPOINT ["python", "scripts/run_pipeline.py"]
CMD ["--config", "configs/banking.yaml"]
