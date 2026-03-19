FROM python:3.11-slim

WORKDIR /app

# System deps for guitarpro / mido
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Python source
COPY python/ ./python/

# Pre-built frontend (build it locally first: cd frontend && ./run -p)
COPY frontend/runtime/dist/build/ ./frontend/runtime/dist/build/

ENV STATIC_DIR=/app/frontend/runtime/dist/build
ENV VITERBI_WORKERS=2

WORKDIR /app/python

EXPOSE 8000

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
