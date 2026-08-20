FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ARG HF_TOKEN
ENV HF_TOKEN=$HF_TOKEN

# Bake the embedding + reranker weights into the image so cold starts on
# free-tier PaaS don't download ~180MB at request time (which times out).
# Pass HF_TOKEN as a Render build-time env var for authenticated (faster) pulls.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')" \
 && python -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/ms-marco-MiniLM-L6-v2')"

EXPOSE 8000

# NOTE: vectors live in the Pinecone cloud index (no local index data).
# Pass PINECONE_API_KEY and GROQ_API_KEY at runtime via --env-file .env.
# --proxy-headers --forwarded-allow-ips "*" makes the rate limiter see the real
# client IP from X-Forwarded-For behind cloud load balancers (Render/Railway).
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--proxy-headers", "--forwarded-allow-ips", "*"]