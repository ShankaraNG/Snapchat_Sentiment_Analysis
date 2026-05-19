# Stage 1: Install dependencies using a slim python base
FROM python:3.11-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir poetry>=2.0.0
RUN poetry config virtualenvs.create false

COPY pyproject.toml ./

RUN poetry install --no-interaction --no-ansi --no-root \
    && rm -rf ~/.cache/pypoetry/cache/ \
    && rm -rf ~/.cache/pip

# Stage 2: Final lightweight runtime container
FROM python:3.11-slim AS runtime

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Copy python packages cleanly from the builder stage
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Download all required NLTK resources
RUN python -m nltk.downloader stopwords punkt punkt_tab wordnet omw-1.4

# Copy your explicit structural pipeline folders
COPY app/ ./app/
COPY artifacts/ ./artifacts/
COPY config/ ./config/
COPY data/ ./data/
COPY ml_build/ ./ml_build/
COPY model/ ./model/

RUN mkdir -p logs

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]