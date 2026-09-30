#failure case review, runs the trained model on the test set, matches predictions
#to ground truth via iou, and visualizes the highest confidence false positives so
#we can eyeball what's actually tripping the model up
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

#minimum iou for a prediction to count as matching a ground truth box of the same class
IOU_MATCH_THRESHOLD = 0.5


#reads one yolo label file into a list of boxes, empty/missing files mean background only
#input: (label_path - path to a single .txt label file)
#returns: list of (class_id, x_center, y_center, width, height) tuples, all normalized 0-1
def parse_labels(label_path: Path):
    if not label_path.exists() or not label_path.stat().st_size:
        return []
    boxes = []
    for line in label_path.read_text().strip().splitlines():
        cls, xc, yc, w, h = line.split()
        boxes.append((int(cls), float(xc), float(yc), float(w), float(h)))
    return boxes


#converts one normalized yolo box into pixel corner coordinates
#input: (box - a (class_id, x_center, y_center, width, height) tuple) (img_w, img_h -
#the image's pixel dimensions)
#returns: (class_id, (x0, y0, x1, y1)), the same class id plus pixel corner coords
def yolo_to_xyxy(box, img_w, img_h):
    cls, xc, yc, w, h = box
    x0 = (xc - w / 2) * img_w
    y0 = (yc - h / 2) * img_h
    x1 = (xc + w / 2) * img_w
    y1 = (yc + h / 2) * img_h
    return cls, (x0, y0, x1, y1)


#standard intersection over union between two boxes in pixel xyxy format
#input: (box_a, box_b - each an (x0, y0, x1, y1) tuple)
#returns: iou as a float 0-1, 0 if there's no overlap
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


#runs the model over every image in the given split, greedily matches predictions to
#ground truth boxes by iou (highest confidence prediction gets first pick of the
#unmatched ground truth boxes of its class), and tallies tp/fp/fn per class
#input: (weights_path - path to the trained .pt weights) (split - "train"/"val"/"test")
#(conf - confidence threshold for predictions to be counted at all)
#returns: list of false positives as (img_path, pred_cls, pred_conf, pred_box_xyxy)
#tuples, sorted highest confidence first
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

        #highest confidence predictions get first pick of ground truth matches
        preds.sort(key=lambda p: -p[2])

        for pred_cls, pred_xyxy, pred_conf in preds:
            #finds the best unmatched, same class ground truth box for this prediction
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

        #any ground truth box nothing matched to is a missed detection
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


#saves a grid of the highest confidence false positives with their predicted box drawn
#on top, so we can actually look at what the model got wrong
#input: (false_positives - list from evaluate()) (n - how many to plot, roughly square
#grid) (out_name - filename to save under OUT_DIR)
#returns: nothing, saves a png as a side effect (or just prints and returns if there
#were no false positives to show)
def visualize_false_positives(false_positives, n=9, out_name="false_positives.png"):
    if not false_positives:
        print("\nNo false positives found, nothing to visualize.")
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

    #blanks out any leftover empty subplots if we had fewer than n false positives
    for ax in axes.flat[len(false_positives[:n]):]:
        ax.axis("off")

    plt.tight_layout()
    out_path = OUT_DIR / out_name
    plt.savefig(out_path, dpi=120)
    print(f"  saved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default=str(ROOT / "model/best.pt"))
    parser.add_argument("--split", default="test")
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()

    fps = evaluate(args.weights, split=args.split, conf=args.conf)
    visualize_false_positives(fps)
