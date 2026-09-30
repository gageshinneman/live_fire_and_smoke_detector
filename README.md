# Live Wildfire Smoke/Fire Detection

A real-time computer vision system that detects wildfire smoke and fire from a live camera feed or video, with a browser UI showing live bounding boxes, a frame-coverage chart, and an event log.

YOLOv8 fine-tuned on the [D-Fire dataset](https://github.com/gaia-solutions-on-demand/DFireDataset), served over a WebSocket for real-time inference.

> **Status:** not currently deployed publicly — run it locally with the steps below.

## Quickstart (Docker)

```bash
git clone https://github.com/gageshinneman/live_fire_and_smoke_detector.git
cd live_fire_and_smoke_detector
docker build -t wildfire-detect .
docker run -p 8000:8000 wildfire-detect
```

Open **http://localhost:8000**. Click **Webcam** (grants camera permission) or **Upload video** to try it on your own footage.

## Quickstart (without Docker)

Requires Python 3.11+. Note: `requirements-backend.txt` pins CPU-only wheels built for the Docker image's Linux environment, so install packages directly here instead — pip will then pick the right build for your OS (including MPS-accelerated torch on Apple Silicon).

```bash
git clone https://github.com/gageshinneman/live_fire_and_smoke_detector.git
cd live_fire_and_smoke_detector
python3 -m venv .venv
source .venv/bin/activate
pip install ultralytics "fastapi[standard]"
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

Open **http://localhost:8000**.

## How it works

```
Browser (webcam/video)
  → JPEG frame, downscaled to 640px
  → WebSocket (/ws/detect)
  → FastAPI backend: YOLOv8 inference (OpenCV decode)
  → JSON detections
  → Canvas overlay + live chart + event log
```

- **Model:** YOLOv8n fine-tuned on D-Fire (21,527 images: smoke + fire, 2 classes). See `scripts/train.py` for the exact training command.
- **Backend:** `backend/main.py` — FastAPI + WebSocket, loads the model once at startup, handles per-frame inference with error handling for malformed frames and client disconnects.
- **Frontend:** `frontend/index.html` — plain HTML/JS + Tailwind CSS, canvas-based bounding-box overlay, a rolling 60s frame-coverage chart, and a debounced event log.

## Model performance

Evaluated on the held-out D-Fire test set:

| | Precision | Recall | mAP@50 | mAP@50-95 |
|---|---|---|---|---|
| **overall** | 0.767 | 0.717 | 0.781 | 0.457 |
| smoke | 0.815 | 0.790 | 0.845 | 0.530 |
| fire | 0.718 | 0.644 | 0.718 | 0.384 |

Fire (smaller, more localized objects) scores lower than smoke (larger, diffuse plumes) — see `scripts/review_failures.py` for the failure-case analysis behind this gap.

## Other scripts

- `scripts/explore_data.py` — dataset class balance, image size distribution, sample visualization
- `scripts/review_failures.py` — runs the model on the test set, flags and visualizes false positives
- `scripts/live_infer.py` — local webcam/video inference loop with an FPS counter, no backend needed
- `scripts/test_segmentation.py` — exploratory YOLOv8 + FastSAM two-stage segmentation benchmark (not currently wired into the live app — see comments in the script for why)

## License / data

D-Fire is used under its own license; see the dataset repo linked above. The bundled demo video (if present locally) is excluded from this repo.
