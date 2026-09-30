# live wildfire smoke/fire detection

real time computer vision system that detects wildfire smoke and fire from a live camera feed or video, with a browser ui showing live bounding boxes, a frame coverage chart, and an event log.

YOLOv8 fine tuned on the [D-Fire dataset](https://github.com/gaia-solutions-on-demand/DFireDataset), served over a websocket for real time inference.

not deployed publicly right now, run it locally with the steps below.

## quickstart (docker)

```bash
git clone https://github.com/gageshinneman/live_fire_and_smoke_detector.git
cd live_fire_and_smoke_detector
docker build -t wildfire-detect .
docker run -p 8000:8000 wildfire-detect
```

open http://localhost:8000, click webcam (grants camera permission) or upload video to try it on your own footage.

## quickstart (without docker)

requires python 3.11+. `requirements-backend.txt` pins cpu only wheels built for the docker image's linux environment, so skip that file here and just install the packages directly, pip will pick the right build for your os, including mps accelerated torch on apple silicon.

```bash
git clone https://github.com/gageshinneman/live_fire_and_smoke_detector.git
cd live_fire_and_smoke_detector
python3 -m venv .venv
source .venv/bin/activate
pip install ultralytics "fastapi[standard]"
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

open http://localhost:8000.

## how it works

```
Browser (webcam/video)
  → JPEG frame, downscaled to 640px
  → WebSocket (/ws/detect)
  → FastAPI backend: YOLOv8 inference (OpenCV decode)
  → JSON detections
  → Canvas overlay + live chart + event log
```

model: YOLOv8n fine tuned on D-Fire (21,527 images, smoke + fire, 2 classes). `scripts/train.py` has the exact training command.

backend: `backend/main.py`, FastAPI + websocket, loads the model once at startup, handles per frame inference with error handling for malformed frames and client disconnects.

frontend: `frontend/index.html`, plain html/js + Tailwind CSS, canvas based bounding box overlay, a rolling 60s frame coverage chart, and a debounced event log.

## model performance

evaluated on the held out D-Fire test set:

| | precision | recall | mAP@50 | mAP@50-95 |
|---|---|---|---|---|
| overall | 0.767 | 0.717 | 0.781 | 0.457 |
| smoke | 0.815 | 0.790 | 0.845 | 0.530 |
| fire | 0.718 | 0.644 | 0.718 | 0.384 |

fire (smaller, more localized objects) scores lower than smoke (larger, diffuse plumes), see `scripts/review_failures.py` for the failure case analysis behind this gap.

## other scripts

- `scripts/explore_data.py`, dataset class balance, image size distribution, sample visualization
- `scripts/review_failures.py`, runs the model on the test set, flags and visualizes false positives
- `scripts/live_infer.py`, local webcam/video inference loop with an fps counter, no backend needed
- `scripts/test_segmentation.py`, exploratory YOLOv8 + FastSAM two stage segmentation benchmark, not wired into the live app, see comments in the script for why

## license / data

D-Fire is used under its own license, see the dataset repo linked above. the bundled demo video, if present locally, is excluded from this repo.
