FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# SEMANTIC_CACHE=true installs sentence-transformers + CPU-only torch and pre-bakes
# the all-MiniLM-L6-v2 embedding model (~500 MB extra).  The resulting image is tagged
# :semantic.  The default (lean) image supports exact-match caching only.
ARG SEMANTIC_CACHE=false

# Pin HuggingFace cache to a predictable path inside /app/data so it can be
# mounted as a volume and shared between builds.
ENV HF_HOME=/app/data/hf_cache

# Install SQLite
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends sqlite3 git \
    && rm -rf /var/lib/apt/lists/*


WORKDIR /app
COPY requirements.txt .
RUN pip install --root-user-action=ignore --no-cache-dir --upgrade pip \
    && pip install --root-user-action=ignore --no-cache-dir -r requirements.txt

# Semantic cache deps — only installed when SEMANTIC_CACHE=true
# CPU-only torch must be installed before sentence-transformers to avoid
# pulling the full CUDA-enabled build (~2.5 GB).
RUN if [ "$SEMANTIC_CACHE" = "true" ]; then \
      pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
      pip install --no-cache-dir sentence-transformers && \
      python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"; \
    fi

# Create database directory and set permissions
RUN mkdir -p /app/data && chown -R www-data:www-data /app/data

COPY . .

RUN chmod +x /app/entrypoint.sh && \
    chown -R www-data:www-data /app

EXPOSE 12434

USER www-data

ENTRYPOINT ["/app/entrypoint.sh"]
