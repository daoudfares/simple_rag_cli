# ============================================================================
# Stage 1 — Builder : compile les dépendances Python
# ============================================================================
FROM python:3.11-slim AS builder

WORKDIR /app

# Dépendances système pour compiler cryptography, snowflake-connector, etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libssl-dev \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ============================================================================
# Stage 2 — Runtime : image finale légère
# ============================================================================
FROM python:3.11-slim

WORKDIR /app

# Copier uniquement les packages Python installés (pas le compilateur)
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copier le code source
COPY pyproject.toml .
COPY app.py .
COPY src/ src/

# Logs temps réel
ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["python", "app.py"]
