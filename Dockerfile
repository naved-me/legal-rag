FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ARG HF_TOKEN
ENV HF_TOKEN=$HF_TOKEN

EXPOSE 8000

# NOTE: vectors live in the Pinecone cloud index (no local index data).
# Pass PINECONE_API_KEY and GROQ_API_KEY at runtime via --env-file .env.
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]