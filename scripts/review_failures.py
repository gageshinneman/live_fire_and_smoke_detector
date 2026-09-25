"""
Phase 2 failure-case review: run the trained model on the test set, match
predictions to ground truth via IoU, and visualize the highest-confidence
false positives (the classic cloud/fog/glare confusions to watch for).
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CLASS_NAMES = ["smoke", "fire"]
OUT_DIR = ROOT / "scripts" / "exploration_output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

IOU_MATCH_THRESHOLD = 0.5


def parse_labels(label_path: Path):
    if not label_path.exists() or not label_path.stat().st_size:
        return []
    boxes = []
    for line in label_path.read_text().strip().splitlines():
        cls, xc, yc, w, h = line.split()
        boxes.append((int(cls), float(xc), float(yc), float(w), float(h)))
    return boxes


def yolo_to_xyxy(box, img_w, img_h):
    cls, xc, yc, w, h = box
    x0 = (xc - w / 2) * img_w
    y0 = (yc - h / 2) * img_h
    x1 = (xc + w / 2) * img_w
    y1 = (yc + h / 2) * img_h
    return cls, (x0, y0, x1, y1)


def iou(box_a, box_b):
    ax0, ay0, ax1, ay1 = box_a
    bx0, by0, bx1, by1 = box_b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
    inter = iw * ih
    area_a = max(0, ax1 - ax0) * max(0, ay1 - ay0)
    area_b = max(0, bx1 - bx0) * max(0, by1 - by0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def evaluate(weights_path, split="test", conf=0.25):
    model = YOLO(weights_path)
    img_dir = DATA / split / "images"
    label_dir = DATA / split / "labels"
    images = sorted(img_dir.glob("*.jpg"))

    results_summary = {c: {"tp": 0, "fp": 0, "fn": 0} for c in range(len(CLASS_NAMES))}
    false_positives = []  # (img_path, pred_cls, pred_conf, pred_box_xyxy)

    for img_path in images:
        label_path = label_dir / (img_path.stem + ".txt")
        gt_boxes = parse_labels(label_path)

        result = model.predict(str(img_path), conf=conf, verbose=False)[0]
        img_w, img_h = result.orig_shape[1], result.orig_shape[0]

        gt_xyxy = [yolo_to_xyxy(b, img_w, img_h) for b in gt_boxes]
        gt_matched = [False] * len(gt_xyxy)

        preds = []
        for box in result.boxes:
            cls = int(box.cls.item())
            xyxy = tuple(box.xyxy[0].tolist())
            confidence = float(box.conf.item())
            preds.append((cls, xyxy, confidence))

        preds.sort(key=lambda p: -p[2])

        for pred_cls, pred_xyxy, pred_conf in preds:
            best_iou, best_idx = 0.0, -1
            for i, (gt_cls, gt_box) in enumerate(gt_xyxy):
                if gt_matched[i] or gt_cls != pred_cls:
                    continue
                cur_iou = iou(pred_xyxy, gt_box)
                if cur_iou > best_iou:
                    best_iou, best_idx = cur_iou, i

            if best_iou >= IOU_MATCH_THRESHOLD:
                gt_matched[best_idx] = True
                results_summary[pred_cls]["tp"] += 1
            else:
                results_summary[pred_cls]["fp"] += 1
                false_positives.append((img_path, pred_cls, pred_conf, pred_xyxy))

        for i, matched in enumerate(gt_matched):
            if not matched:
                results_summary[gt_xyxy[i][0]]["fn"] += 1

    print(f"\n=== Per-class results on '{split}' (conf={conf}, IoU match={IOU_MATCH_THRESHOLD}) ===")
    for cls_id, name in enumerate(CLASS_NAMES):
        s = results_summary[cls_id]
        precision = s["tp"] / (s["tp"] + s["fp"]) if (s["tp"] + s["fp"]) else 0
        recall = s["tp"] / (s["tp"] + s["fn"]) if (s["tp"] + s["fn"]) else 0
        print(
            f"  {name:5s}: TP={s['tp']:4d} FP={s['fp']:4d} FN={s['fn']:4d} "
            f"precision={precision:.3f} recall={recall:.3f}"
        )

    false_positives.sort(key=lambda fp: -fp[2])
    return false_positives


def visualize_false_positives(false_positives, n=9, out_name="false_positives.png"):
    if not false_positives:
        print("\nNo false positives found — nothing to visualize.")
        return

    print(f"\n=== Saving top {min(n, len(false_positives))} highest-confidence false positives ===")
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    colors = {0: "yellow", 1: "red"}

    for ax, (img_path, cls, conf, xyxy) in zip(axes.flat, false_positives[:n]):
        with Image.open(img_path) as im:
            ax.imshow(im)
        x0, y0, x1, y1 = xyxy
        rect = plt.Rectangle(
            (x0, y0), x1 - x0, y1 - y0,
            linewidth=2, edgecolor=colors.get(cls, "white"), facecolor="none"
        )
        ax.add_patch(rect)
        ax.set_title(f"{img_path.stem}\n{CLASS_NAMES[cls]} conf={conf:.2f}", fontsize=8)
        ax.axis("off")

    for ax in axes.flat[len(false_positives[:n]):]:
        ax.axis("off")

    plt.tight_layout()
    out_path = OUT_DIR / out_name
    plt.savefig(out_path, dpi=120)
    print(f"  saved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default=str(ROOT / "runs/detect/runs/train/baseline-3/weights/best.pt"))
    parser.add_argument("--split", default="test")
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()

    fps = evaluate(args.weights, split=args.split, conf=args.conf)
    visualize_false_positives(fps)
