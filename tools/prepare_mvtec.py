# -*- coding: utf-8 -*-
"""Build the metadata index for MVTec-AD.

The pipeline reads a single ``meta_mask_box.json`` that lists every test image
and, for anomalous images, the ground-truth mask plus a small per-mask JSON file
holding the box(es) derived from that mask. This script produces all of it from
a raw MVTec-AD directory.

Expected input layout (the official release)::

    <root>/<category>/train/good/*.png
    <root>/<category>/test/good/*.png
    <root>/<category>/test/<defect>/*.png
    <root>/<category>/ground_truth/<defect>/*_mask.png

Output::

    <root>/<category>/ground_truth/<defect>/<name>_mask_box.json
    <root>/meta_mask_box.json

Usage:
    python tools/prepare_mvtec.py --dataset_root /path/to/mvtec_anomaly_detection
"""

import argparse
import json
import os

import cv2
import numpy as np
from PIL import Image

MVTEC_CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid",
    "hazelnut", "leather", "metal_nut", "pill", "screw",
    "tile", "toothbrush", "transistor", "wood", "zipper",
]


def mask_to_boxes(mask_path, min_area=10):
    """Extract bounding boxes of all connected components of a GT mask."""
    mask = np.array(Image.open(mask_path).convert("L")) > 0
    mask = (mask.astype(np.uint8)) * 255
    height, width = mask.shape

    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        boxes.append([x, y, x + w, y + h])
    return boxes, {"height": height, "width": width}


def main():
    parser = argparse.ArgumentParser(description="Build meta_mask_box.json for MVTec-AD")
    parser.add_argument("--dataset_root", required=True, help="MVTec-AD root directory")
    parser.add_argument("--output", default="", help="Output meta path (default: <root>/meta_mask_box.json)")
    parser.add_argument("--categories", nargs="*", default=MVTEC_CATEGORIES)
    parser.add_argument("--min_area", type=int, default=10, help="Ignore mask components smaller than this")
    args = parser.parse_args()

    root = args.dataset_root
    output_path = args.output or os.path.join(root, "meta_mask_box.json")

    meta = {"train": {}, "test": {}}
    num_boxes = 0

    for category in args.categories:
        cat_dir = os.path.join(root, category)
        if not os.path.isdir(cat_dir):
            print(f"skip missing category: {category}")
            continue

        # ---------------- train / test-good ----------------
        meta["train"][category] = []
        meta["test"][category] = []

        train_good = os.path.join(cat_dir, "train", "good")
        if os.path.isdir(train_good):
            for name in sorted(os.listdir(train_good)):
                if not name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif")):
                    continue
                meta["train"][category].append({
                    "img_path": f"{category}/train/good/{name}",
                    "mask_path": "",
                    "cls_name": category,
                    "specie_name": "good",
                    "anomaly": 0,
                    "box_json_path": "",
                })

        test_dir = os.path.join(cat_dir, "test")
        for defect in sorted(os.listdir(test_dir)) if os.path.isdir(test_dir) else []:
            defect_dir = os.path.join(test_dir, defect)
            if not os.path.isdir(defect_dir):
                continue
            is_normal = defect == "good"

            for name in sorted(os.listdir(defect_dir)):
                if not name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif")):
                    continue
                img_rel = f"{category}/test/{defect}/{name}"

                if is_normal:
                    meta["test"][category].append({
                        "img_path": img_rel,
                        "mask_path": "",
                        "cls_name": category,
                        "specie_name": "good",
                        "anomaly": 0,
                        "box_json_path": "",
                    })
                    continue

                stem = os.path.splitext(name)[0]
                mask_rel = f"{category}/ground_truth/{defect}/{stem}_mask.png"
                mask_abs = os.path.join(root, mask_rel)
                if not os.path.exists(mask_abs):
                    print(f"warning: mask not found for {img_rel}, skipped")
                    continue

                boxes, image_size = mask_to_boxes(mask_abs, args.min_area)
                box_rel = f"{category}/ground_truth/{defect}/{stem}_mask_box.json"
                box_abs = os.path.join(root, box_rel)
                with open(box_abs, "w", encoding="utf-8") as f:
                    json.dump(
                        {"img_path": img_rel, "image_size": image_size, "gt_boxes": boxes},
                        f, indent=4,
                    )
                num_boxes += 1

                meta["test"][category].append({
                    "img_path": img_rel,
                    "mask_path": mask_rel,
                    "cls_name": category,
                    "specie_name": defect,
                    "anomaly": 1,
                    "box_json_path": box_rel,
                })

        n_test = len(meta["test"][category])
        n_anom = sum(item["anomaly"] for item in meta["test"][category])
        print(f"{category:12s} test={n_test:4d} (anomaly={n_anom:4d})")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=4)

    total_test = sum(len(v) for v in meta["test"].values())
    total_anom = sum(sum(i["anomaly"] for i in v) for v in meta["test"].values())
    print(f"\nwrote {output_path}")
    print(f"test images: {total_test} (anomaly={total_anom}), box json files: {num_boxes}")


if __name__ == "__main__":
    main()
