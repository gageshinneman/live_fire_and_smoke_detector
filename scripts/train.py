#trains the yolov8 baseline on d-fire
#wraps the same run used to produce model/best.pt:
#  yolo detect train model=yolov8n.pt data=data.yaml epochs=100 patience=15
#    imgsz=640 batch=16 device=mps project=runs/train name=baseline
#re-run as is to reproduce the baseline, or override any argument on the command line
#(e.g. --model yolov8s.pt --epochs 50) to try a variant
import argparse
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--data", default=str(ROOT / "data.yaml"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)   #early stop after this many epochs with no val improvement
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--project", default=str(ROOT / "runs/train"))
    parser.add_argument("--name", default="baseline")
    args = parser.parse_args()

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        patience=args.patience,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
    )
