# Backend image for the wildfire smoke/fire detection service.
# Serves the FastAPI + WebSocket inference API and the static frontend.

FROM python:3.11-slim

# libgl1/libglib2.0-0: required by opencv-python (a dependency of ultralytics)
# even in headless/server use, since it links against them at import time.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements-backend.txt .
RUN pip install --no-cache-dir -r requirements-backend.txt

COPY backend/ backend/
COPY frontend/ frontend/
COPY model/ model/

ENV MODEL_PATH=/app/model/best.pt

EXPOSE 8000

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
