FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DATA_DIR=/app/data

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py .
COPY agentic_platform ./agentic_platform
RUN mkdir -p /app/data
EXPOSE 8000
CMD ["python", "app.py", "serve", "--host", "0.0.0.0", "--port", "8000"]
