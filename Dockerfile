FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ARG HF_TOKEN
ENV HF_TOKEN=$HF_TOKEN

# Build the local vector index at image build time.
RUN python ingest.py

EXPOSE 8000

# NOTE: with multiple workers each process loads its own model copies
# and, until a Redis store is wired in, sessions live in SQLite (shared).
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]