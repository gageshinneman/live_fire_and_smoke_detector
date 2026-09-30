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

# Caps the underlying BLAS/OpenMP thread pools before torch/opencv initialize
# them at import time, so they never oversubscribe a small/shared cloud CPU.
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1

EXPOSE 8000

# Render (and most PaaS hosts) inject a PORT env var the app must bind to;
# ${PORT:-8000} falls back to 8000 for local `docker run` where it's unset.
CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
