# --- STAGE 1: Builder ---
FROM python:3.11-slim AS builder

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y wget

# Install requirements into a local folder
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --user -r requirements.txt

# Download weights
RUN wget https://pjreddie.com/media/files/yolov3.weights -O yolov3.weights

# --- STAGE 2: Final Image ---
FROM python:3.11-slim

WORKDIR /app
# Copy only the installed packages from the builder
COPY --from=builder /root/.local /root/.local
COPY --from=builder /app/yolov3.weights /app/yolov3.weights
COPY . .

# Update PATH to find the installed packages
ENV PATH=/root/.local/bin:$PATH
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install minimal system libs for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

EXPOSE 8050
CMD ["python", "app.py"]