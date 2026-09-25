"""
Phase 4: FastAPI backend for live wildfire smoke/fire detection.

Loads the trained YOLOv8 model once at startup and exposes a WebSocket
endpoint that accepts JPEG-encoded frames and returns detection results
as JSON. A WebSocket is used instead of plain HTTP request/response per
frame to avoid per-request overhead and keep the loop fast enough to
feel real-time.
"""
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS_PATH = Path(os.environ.get("MODEL_PATH", ROOT / "model" / "best.pt"))
CLASS_NAMES = ["smoke", "fire"]
DEFAULT_CONF = 0.25

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("wildfire-backend")

model_state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Loading model from {WEIGHTS_PATH} ...")
    model_state["model"] = YOLO(str(WEIGHTS_PATH))
    logger.info("Model loaded.")
    yield
    model_state.clear()


app = FastAPI(lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "model" in model_state}


def run_inference(frame: np.ndarray, conf: float = DEFAULT_CONF) -> list[dict]:
    model = model_state["model"]
    result = model.predict(frame, conf=conf, verbose=False)[0]
    detections = []
    for box in result.boxes:
        cls = int(box.cls.item())
        x0, y0, x1, y1 = box.xyxy[0].tolist()
        detections.append({
            "class_id": cls,
            "class_name": CLASS_NAMES[cls] if cls < len(CLASS_NAMES) else str(cls),
            "confidence": round(float(box.conf.item()), 4),
            "box": {"x0": x0, "y0": y0, "x1": x1, "y1": y1},
        })
    return detections


@app.websocket("/ws/detect")
async def websocket_detect(websocket: WebSocket):
    await websocket.accept()
    client = websocket.client
    logger.info(f"Client connected: {client}")

    frame_count = 0
    try:
        while True:
            data = await websocket.receive_bytes()
            frame_count += 1

            frame = None
            if data:
                frame_array = np.frombuffer(data, dtype=np.uint8)
                try:
                    frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
                except cv2.error:
                    frame = None

            if frame is None:
                await websocket.send_json({
                    "error": "bad_frame",
                    "message": "Could not decode frame as an image.",
                    "frame_number": frame_count,
                })
                continue

            t0 = time.perf_counter()
            try:
                detections = run_inference(frame)
            except Exception as e:
                logger.exception("Inference failed")
                await websocket.send_json({
                    "error": "inference_error",
                    "message": str(e),
                    "frame_number": frame_count,
                })
                continue
            inference_ms = (time.perf_counter() - t0) * 1000

            await websocket.send_json({
                "frame_number": frame_count,
                "detections": detections,
                "inference_ms": round(inference_ms, 2),
            })

    except WebSocketDisconnect:
        logger.info(f"Client disconnected: {client} (processed {frame_count} frames)")
    except Exception:
        logger.exception(f"Unexpected error on connection {client}")
        try:
            await websocket.close()
        except RuntimeError:
            pass


# Mounted last: serves the frontend at "/" so it shares an origin with the
# WebSocket endpoint above, avoiding CORS entirely.
app.mount("/", StaticFiles(directory=str(ROOT / "frontend"), html=True), name="frontend")
