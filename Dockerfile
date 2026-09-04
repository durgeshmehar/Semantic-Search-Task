# CPU-only image: no CUDA, which keeps this a few hundred MB rather than several GB.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    SENTENCE_TRANSFORMERS_HOME=/opt/models

WORKDIR /app

# Torch pulls OpenMP; build tools are not needed since every wheel is prebuilt.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

# Install the CPU torch wheel explicitly. Without this, sentence-transformers
# drags in the default CUDA build -- ~2.5 GB of GPU libraries this service
# never uses.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.5.1

COPY requirements.txt .
RUN pip install -r requirements.txt

# Bake the embedding model into the image so startup needs no network and the
# first search isn't waiting on a download.
RUN python -c "\
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', \
cache_folder='/opt/models')"

# Only now pin offline mode: the weights are cached above, so runtime never
# reaches for the network. Set earlier, this would have blocked that download.
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

COPY app ./app
COPY tests ./tests

# Uploads, indexes and the database live here; mounted as a volume in compose
# so they survive container restarts.
RUN mkdir -p /app/data
ENV DATA_DIR=/app/data

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

# One worker process: the indexing thread pool and its SQLite queue live
# in-process, so multiple uvicorn workers would each start their own pool.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
