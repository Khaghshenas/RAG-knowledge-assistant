FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple

# Copy the rest of the application code
COPY src/ ./src/
COPY app/ ./app/
COPY data/processed/faiss.index /app/data/processed/
COPY data/processed/faiss_metadata.pkl /app/data/processed/
COPY data/processed/chunks.jsonl /app/data/processed/
COPY config.yaml .

# Create a directory for the local vector database
RUN mkdir -p /app/data/processed /app/app

# Expose the port
EXPOSE 8000

CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]