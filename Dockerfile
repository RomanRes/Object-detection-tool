# --- STAGE 1: Builder ---
FROM python:3.11-slim AS builder

WORKDIR /app

# virtuelle Umgebung statt --user
RUN python -m venv /venv
ENV PATH="/venv/bin:$PATH"

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# weights laden
RUN apt-get update && apt-get install -y wget && \
    wget https://pjreddie.com/media/files/yolov3.weights -O yolov3.weights && \
    rm -rf /var/lib/apt/lists/*

# --- STAGE 2: Final Image ---
FROM python:3.11-slim

WORKDIR /app

# nur saubere venv kopieren
COPY --from=builder /venv /venv
ENV PATH="/venv/bin:$PATH"

# weights kopieren
COPY --from=builder /app/yolov3.weights /app/yolov3.weights

# code
COPY . .

EXPOSE 8050

CMD ["python", "app.py"]
#