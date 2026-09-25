"""
Phase 6.5 exploration: detect with YOLOv8, then box-prompt FastSAM within
each detection to generate a pixel-level mask. Visualizes results and
times the added latency, before deciding whether this is viable live.
"""
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from ultralytics import FastSAM, YOLO

ROOT = Path(__file__).resolve().parent.parent
DET_WEIGHTS = ROOT / "runs/detect/runs/train/baseline-3/weights/best.pt"
SEG_WEIGHTS = "FastSAM-s.pt"  # auto-downloaded by ultralytics on first use
CLASS_NAMES = ["smoke", "fire"]
COLORS = [(230, 198, 25), (248, 81, 73)]  # smoke=yellow, fire=red (RGB)
OUT_DIR = ROOT / "scripts" / "exploration_output"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def pick_sample_images(n=6):
    """Grab test images that have at least one smoke detection ground truth,
    since smoke is the harder case for mask quality (diffuse, low-contrast)."""
    label_dir = ROOT / "data" / "test" / "labels"
    img_dir = ROOT / "data" / "test" / "images"
    picks = []
    for lf in sorted(label_dir.glob("*.txt")):
        if not lf.stat().st_size:
            continue
        text = lf.read_text()
        if text.strip().split()[0] == "0":  # starts with a smoke box
            picks.append(img_dir / (lf.stem + ".jpg"))
        if len(picks) >= n:
            break
    return picks


def run_pipeline(det_model, seg_model, img_path, conf=0.25):
    result = det_model.predict(str(img_path), conf=conf, verbose=False)[0]
    boxes = [box.xyxy[0].tolist() for box in result.boxes]
    classes = [int(box.cls.item()) for box in result.boxes]
    confs = [float(box.conf.item()) for box in result.boxes]

    if not boxes:
        return boxes, classes, confs, [], 0.0

    t0 = time.perf_counter()
    seg_result = seg_model.predict(str(img_path), bboxes=boxes, verbose=False)[0]
    seg_ms = (time.perf_counter() - t0) * 1000

    masks = seg_result.masks.data.cpu().numpy() if seg_result.masks is not None else []
    return boxes, classes, confs, masks, seg_ms


def overlay_masks(img_array, masks, classes):
    overlay = img_array.copy()
    for mask, cls in zip(masks, classes):
        mask_resized = np.array(Image.fromarray(mask).resize((img_array.shape[1], img_array.shape[0])))
        color = np.array(COLORS[cls] if cls < len(COLORS) else (255, 255, 255))
        overlay[mask_resized > 0.5] = (
            0.5 * overlay[mask_resized > 0.5] + 0.5 * color
        ).astype(np.uint8)
    return overlay


def main():
    print("Loading detection model...")
    det_model = YOLO(str(DET_WEIGHTS))
    print("Loading FastSAM (downloads on first use)...")
    seg_model = FastSAM(SEG_WEIGHTS)

    images = pick_sample_images(n=6)
    print(f"Testing on {len(images)} images with smoke ground truth...")

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    seg_times = []

    for ax, img_path in zip(axes.flat, images):
        boxes, classes, confs, masks, seg_ms = run_pipeline(det_model, seg_model, img_path)
        seg_times.append(seg_ms)

        with Image.open(img_path) as im:
            img_array = np.array(im.convert("RGB"))

        if len(masks):
            img_array = overlay_masks(img_array, masks, classes)

        ax.imshow(img_array)
        for box, cls, conf in zip(boxes, classes, confs):
            x0, y0, x1, y1 = box
            color = np.array(COLORS[cls] if cls < len(COLORS) else (255, 255, 255)) / 255
            rect = plt.Rectangle((x0, y0), x1 - x0, y1 - y0, linewidth=1.5, edgecolor=color, facecolor="none")
            ax.add_patch(rect)
        ax.set_title(f"{img_path.stem}\nseg time: {seg_ms:.0f}ms, {len(masks)} masks", fontsize=9)
        ax.axis("off")

    plt.tight_layout()
    out_path = OUT_DIR / "segmentation_test.png"
    plt.savefig(out_path, dpi=120)
    print(f"\nSaved: {out_path}")
    print(f"Avg FastSAM segmentation time: {sum(seg_times)/len(seg_times):.1f}ms per image "
          f"({len(seg_times)} images, boxes varying)")


if __name__ == "__main__":
    main()
