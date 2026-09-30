#dataset exploration, class balance, image size distribution, sample visualization
#for the d-fire dataset (yolo format: data/{train,val,test}/{images,labels})
import random
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
SPLITS = ["train", "val", "test"]
CLASS_NAMES = ["smoke", "fire"]
OUT_DIR = ROOT / "scripts" / "exploration_output"
OUT_DIR.mkdir(parents=True, exist_ok=True)


#reads one yolo label file into a list of boxes, empty files mean background only
#input: (label_path - path to a single .txt label file)
#returns: list of (class_id, x_center, y_center, width, height) tuples, all normalized 0-1
def parse_labels(label_path: Path):
    if not label_path.stat().st_size:
        return []
    boxes = []
    for line in label_path.read_text().strip().splitlines():
        cls, xc, yc, w, h = line.split()
        boxes.append((int(cls), float(xc), float(yc), float(w), float(h)))
    return boxes


#prints per split box counts and background-only image counts, per class
#input: none, reads DATA/SPLITS/CLASS_NAMES globals
#returns: nothing, prints only
def class_balance():
    print("\n=== Class balance (per box, and per image) ===")
    for split in SPLITS:
        label_dir = DATA / split / "labels"
        label_files = list(label_dir.glob("*.txt"))

        box_counts = Counter()
        image_has_class = Counter()
        n_background = 0

        for lf in label_files:
            boxes = parse_labels(lf)
            if not boxes:
                n_background += 1
                continue
            #tracks which classes show up in this image, so a class with multiple
            #boxes in one image still only counts once for "images containing X"
            classes_in_image = set()
            for cls, *_ in boxes:
                box_counts[cls] += 1
                classes_in_image.add(cls)
            for cls in classes_in_image:
                image_has_class[cls] += 1

        n_images = len(label_files)
        print(f"\n-- {split} ({n_images} images) --")
        print(f"  background-only images: {n_background} ({n_background/n_images:.1%})")
        for cls_id, name in enumerate(CLASS_NAMES):
            print(
                f"  {name:5s}: {box_counts[cls_id]:6d} boxes across "
                f"{image_has_class[cls_id]:6d} images "
                f"({image_has_class[cls_id]/n_images:.1%} of images)"
            )


#samples images per split and prints width/height stats, to sanity check what
#imgsz/resize choice makes sense for training
#input: (sample_size - max images to sample per split)
#returns: nothing, prints only
def image_size_distribution(sample_size=1000):
    print("\n=== Image size distribution (sampled) ===")
    for split in SPLITS:
        img_dir = DATA / split / "images"
        images = list(img_dir.glob("*.jpg"))
        sample = random.sample(images, min(sample_size, len(images)))

        sizes = []
        for img_path in sample:
            with Image.open(img_path) as im:
                sizes.append(im.size)  # (width, height)

        widths = [s[0] for s in sizes]
        heights = [s[1] for s in sizes]
        unique_sizes = Counter(sizes)

        print(f"\n-- {split} (sampled {len(sample)}) --")
        print(f"  width:  min={min(widths)} max={max(widths)} mean={sum(widths)/len(widths):.0f}")
        print(f"  height: min={min(heights)} max={max(heights)} mean={sum(heights)/len(heights):.0f}")
        print(f"  unique (w,h) pairs: {len(unique_sizes)}")
        print(f"  most common: {unique_sizes.most_common(3)}")


#grabs a handful of labeled training images and saves a grid with boxes drawn on
#top, for a quick eyeball check that labels line up with what's actually in frame
#input: (n - how many sample images to draw, plotted as a roughly square grid)
#returns: nothing, saves a png to OUT_DIR as a side effect
def visualize_samples(n=9):
    print(f"\n=== Saving {n} annotated sample images to {OUT_DIR} ===")
    img_dir = DATA / "train" / "images"
    label_dir = DATA / "train" / "labels"

    non_empty_labels = [lf for lf in label_dir.glob("*.txt") if lf.stat().st_size > 0]
    sample_labels = random.sample(non_empty_labels, min(n, len(non_empty_labels)))

    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    colors = {0: "yellow", 1: "red"}  # smoke, fire

    for ax, lf in zip(axes.flat, sample_labels):
        img_path = img_dir / (lf.stem + ".jpg")
        with Image.open(img_path) as im:
            w, h = im.size
            ax.imshow(im)

        #yolo boxes are normalized center x/y + width/height, convert to pixel
        #top left corner + pixel width/height for matplotlib's Rectangle
        for cls, xc, yc, bw, bh in parse_labels(lf):
            box_w, box_h = bw * w, bh * h
            x0 = xc * w - box_w / 2
            y0 = yc * h - box_h / 2
            rect = plt.Rectangle(
                (x0, y0), box_w, box_h,
                linewidth=2, edgecolor=colors.get(cls, "white"), facecolor="none"
            )
            ax.add_patch(rect)
            ax.text(x0, y0 - 5, CLASS_NAMES[cls], color=colors.get(cls, "white"), fontsize=8)

        ax.set_title(lf.stem, fontsize=8)
        ax.axis("off")

    plt.tight_layout()
    out_path = OUT_DIR / "sample_annotations.png"
    plt.savefig(out_path, dpi=120)
    print(f"  saved: {out_path}")


if __name__ == "__main__":
    random.seed(0)   #for reproducible sampling
    class_balance()
    image_size_distribution()
    visualize_samples()
