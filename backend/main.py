#FastAPI backend for live wildfire smoke/fire detection
#loads the trained YOLOv8 model once at startup and exposes a websocket endpoint that
#accepts jpeg encoded frames and returns detection results as json. websocket is used
#instead of plain http request/response per frame to avoid per request overhead and
#keep the loop fast enough to feel real time
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

#env var lets the docker image point at a different weights path, defaults to the
#trained model checked into model/best.pt for local runs
WEIGHTS_PATH = Path(os.environ.get("MODEL_PATH", ROOT / "model" / "best.pt"))
CLASS_NAMES = ["smoke", "fire"]
DEFAULT_CONF = 0.25

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("wildfire-backend")

#holds the loaded model, set once in lifespan() at startup so every request/connection
#reuses the same in memory model instead of reloading it
model_state = {}


#loads the model once when the app starts, and clears it on shutdown
#input: (app - the FastAPI app instance, unused but required by the decorator's signature)
#returns: nothing, yields control back to FastAPI while the app runs
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Loading model from {WEIGHTS_PATH} ...")
    model_state["model"] = YOLO(str(WEIGHTS_PATH))
    logger.info("Model loaded.")
    yield
    model_state.clear()


app = FastAPI(lifespan=lifespan)


#simple healthcheck endpoint, used by deploy platforms and for manual testing
#input: none
#returns: dict with status and whether the model has finished loading
@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "model" in model_state}


#runs the loaded model on a single decoded frame and formats the boxes as plain dicts
#input: (frame - a decoded bgr image, numpy array) (conf - confidence threshold to filter
#detections)
#returns: list of detection dicts, one per box, each with class_id/class_name/confidence/box
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


#main websocket loop, one connection per client. client sends raw jpeg bytes per frame,
#we decode/run inference/reply with detections as json, and keep looping until they
#disconnect. handles bad frames and model errors without killing the connection
#input: (websocket - the accepted connection)
#returns: nothing, runs until the client disconnects or an unexpected error happens
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

            #decode the incoming jpeg bytes into a frame, catching both an empty
            #payload and cv2's own decode failure (opencv raises rather than
            #returning None for some malformed inputs, e.g. empty buffers)
            frame = None
            if data:
                frame_array = np.frombuffer(data, dtype=np.uint8)
                try:
                    frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
                except cv2.error:
                    frame = None

            #bad frame, report it and keep the connection alive for the next one
            if frame is None:
                await websocket.send_json({
                    "error": "bad_frame",
                    "message": "Could not decode frame as an image.",
                    "frame_number": frame_count,
                })
                continue

            #runs inference and times it, reports any model error the same way as
            #a bad frame instead of dropping the connection
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

    #normal client disconnect, not an error
    except WebSocketDisconnect:
        logger.info(f"Client disconnected: {client} (processed {frame_count} frames)")
    #anything else unexpected, log it and try to close cleanly
    except Exception:
        logger.exception(f"Unexpected error on connection {client}")
        try:
            await websocket.close()
        except RuntimeError:
            pass


#mounted last, serves the frontend at "/" so it shares an origin with the websocket
#endpoint above, avoiding CORS entirely
app.mount("/", StaticFiles(directory=str(ROOT / "frontend"), html=True), name="frontend")
