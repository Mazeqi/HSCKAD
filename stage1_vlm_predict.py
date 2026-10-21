# -*- coding: utf-8 -*-
"""Stage 1: VLM anomaly localization with hierarchical concept prompting.

For every test image of MVTec-AD the VLM (Qwen3-VL) is asked to output the
anomalous regions of the category. The raw detections are post-processed with
the *physical calibration* (PC) constraints stored in
``configs/mvtec/category_config.json``:

1. confidence threshold,
2. class-agnostic NMS,
3. plausible-area band ``[min_area_ratio, max_area_ratio]``.

The surviving detections are written to
``<output_dir>/<relative image dir>/<image stem>_qwen.json`` in the
``[{"label", "bbox_2d", "confidence"}, ...]`` format with ``bbox_2d``
normalized to ``[0, 1000]``. Stage 2 consumes exactly these files.

Usage:
    python stage1_vlm_predict.py \
        --meta_json       /path/to/mvtec/meta_mask_box.json \
        --dataset_root    /path/to/mvtec \
        --checkpoint      Qwen/Qwen3-VL-2B-Instruct \
        --category_templates configs/mvtec/category_templates.json \
        --category_config    configs/mvtec/category_config.json \
        --defect_types       configs/mvtec/defect_types.json \
        --output_dir      outputs/mvtec_vlm
"""

import argparse
import json
import os
from datetime import datetime

from PIL import Image
import torch
from torchvision.ops import nms
from tqdm import tqdm

from method.qwen_vlm import QwenVLInferencer

DEFAULT_CONFIG = {
    "confidence_threshold": 0.2,
    "temperature": 0.2,
    "top_p": 0.9,
    "top_k": -1,
    "max_tokens": 1024,
    "nms_iou": 0.4,
    "min_area_ratio": 0.0005,
    "max_area_ratio": 0.6,
}


# ---------------------------------------------------------------------------
# box utilities
# ---------------------------------------------------------------------------
def box_iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def match_boxes(pred_boxes, gt_boxes, iou_threshold):
    """Greedy one-to-one matching; returns (tp, fp, fn, sum_iou)."""
    matched = set()
    tp, fp, sum_iou = 0, 0, 0.0
    for p in pred_boxes:
        best_iou, best_idx = 0.0, -1
        for i, g in enumerate(gt_boxes):
            if i in matched:
                continue
            iou = box_iou(p, g)
            if iou > best_iou:
                best_iou, best_idx = iou, i
        if best_iou >= iou_threshold:
            tp += 1
            matched.add(best_idx)
            sum_iou += best_iou
        else:
            fp += 1
    return tp, fp, len(gt_boxes) - len(matched), sum_iou


# ---------------------------------------------------------------------------
# physical calibration
# ---------------------------------------------------------------------------
def calibrate(detections, cfg, img_w, img_h):
    """Confidence filtering -> NMS -> plausible-area band."""
    kept = [
        d for d in detections
        if d.get("confidence", 1.0) >= cfg["confidence_threshold"]
        and isinstance(d.get("bbox_2d"), (list, tuple)) and len(d["bbox_2d"]) == 4
    ]

    if len(kept) > 1:
        keep_idx = nms(
            torch.tensor([d["bbox_2d"] for d in kept], dtype=torch.float32),
            torch.tensor([d.get("confidence", 1.0) for d in kept], dtype=torch.float32),
            iou_threshold=cfg["nms_iou"],
        )
        kept = [kept[i] for i in keep_idx]

    img_area = float(img_w * img_h)
    min_area = cfg["min_area_ratio"] * img_area
    max_area = cfg["max_area_ratio"] * img_area

    filtered = []
    for d in kept:
        bbox = d["bbox_2d"]
        w = (bbox[2] - bbox[0]) / 1000.0 * img_w
        h = (bbox[3] - bbox[1]) / 1000.0 * img_h
        if min_area <= w * h <= max_area:
            filtered.append(d)
    return filtered


def to_pixel_boxes(detections, img_w, img_h):
    boxes = []
    for d in detections:
        b = d["bbox_2d"]
        x1, x2 = sorted((b[0] / 1000.0 * img_w, b[2] / 1000.0 * img_w))
        y1, y2 = sorted((b[1] / 1000.0 * img_h, b[3] / 1000.0 * img_h))
        boxes.append([x1, y1, x2, y2])
    return boxes


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def load_tasks(meta_json_path, dataset_root, defect_types_path):
    with open(meta_json_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    with open(defect_types_path, "r", encoding="utf-8") as f:
        defect_types = json.load(f)

    tasks = []
    for split, categories in meta.items():
        if split != "test":
            continue
        for category, items in categories.items():
            for item in items:
                tasks.append({
                    "category": category,
                    "anomaly": int(item.get("anomaly", 0)),
                    "defect_types": defect_types.get(category, []),
                    "img_path_rel": item["img_path"],
                    "img_path_abs": os.path.join(dataset_root, item["img_path"]),
                    "box_json_path_abs": (
                        os.path.join(dataset_root, item["box_json_path"])
                        if item.get("box_json_path") else ""
                    ),
                })
    return tasks


def main():
    parser = argparse.ArgumentParser(description="Stage 1: VLM anomaly localization")
    parser.add_argument("--meta_json", required=True)
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--checkpoint", default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--category_templates", default="configs/mvtec/category_templates.json")
    parser.add_argument("--category_config", default="configs/mvtec/category_config.json")
    parser.add_argument("--defect_types", default="configs/mvtec/defect_types.json")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--iou_threshold", type=float, default=0.5,
                        help="IoU threshold of the box-level matching used for the report")
    parser.add_argument("--max_model_len", type=int, default=8192)
    parser.add_argument("--max_images", type=int, default=None)
    parser.add_argument("--save_pred_img", action="store_true", help="Also dump annotated images")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.category_config, "r", encoding="utf-8") as f:
        cat_config = json.load(f)
    default_cfg = dict(DEFAULT_CONFIG)
    default_cfg.update(cat_config.get("default", {}))

    def cfg_for(category):
        cfg = dict(default_cfg)
        cfg.update(cat_config.get(category, {}))
        return cfg

    tasks = load_tasks(args.meta_json, args.dataset_root, args.defect_types)
    if args.max_images:
        tasks = tasks[: args.max_images]
    print(f"Loaded {len(tasks)} test images "
          f"({sum(t['anomaly'] for t in tasks)} anomalous).")

    inferencer = QwenVLInferencer(
        checkpoint_path=args.checkpoint, max_model_len=args.max_model_len
    )
    inferencer.load_category_templates(args.category_templates)

    stats = {c: {"TP": 0, "FP": 0, "FN": 0, "SUM_IOU": 0.0, "GT": 0} for c in
             sorted({t["category"] for t in tasks})}
    global_stats = {"TP": 0, "FP": 0, "FN": 0, "SUM_IOU": 0.0, "GT": 0}
    log_rows = []

    for start in tqdm(range(0, len(tasks), args.batch_size), desc="stage1: VLM"):
        batch = tasks[start: start + args.batch_size]
        cfg = cfg_for(batch[0]["category"])

        messages = [
            inferencer.build_messages(t["img_path_abs"], t["category"], t["defect_types"])
            for t in batch
        ]
        outputs = inferencer.generate(
            messages,
            temperature=cfg["temperature"],
            max_tokens=cfg["max_tokens"],
            top_p=cfg["top_p"],
            top_k=cfg["top_k"],
        )

        for task, text in zip(batch, outputs):
            detections = inferencer.parse_json(text)

            with Image.open(task["img_path_abs"]) as im:
                img_w, img_h = im.size
            task_cfg = cfg_for(task["category"])
            kept = calibrate(detections, task_cfg, img_w, img_h)

            rel_dir = os.path.dirname(task["img_path_rel"])
            stem = os.path.splitext(os.path.basename(task["img_path_rel"]))[0]
            save_dir = os.path.join(args.output_dir, rel_dir)
            os.makedirs(save_dir, exist_ok=True)
            with open(os.path.join(save_dir, f"{stem}_qwen.json"), "w", encoding="utf-8") as f:
                json.dump(kept, f, indent=4, ensure_ascii=False)

            if args.save_pred_img:
                inferencer.draw_bounding_boxes(
                    task["img_path_abs"], kept,
                    os.path.join(save_dir, f"{stem}_qwen.png"),
                )

            if task["anomaly"] == 1 and task["box_json_path_abs"]:
                with open(task["box_json_path_abs"], "r", encoding="utf-8") as f:
                    gt = json.load(f)
                gt_boxes = gt.get("gt_boxes", [])
                pred_boxes = to_pixel_boxes(kept, img_w, img_h)
                tp, fp, fn, sum_iou = match_boxes(pred_boxes, gt_boxes, args.iou_threshold)
                for bucket in (stats[task["category"]], global_stats):
                    bucket["TP"] += tp
                    bucket["FP"] += fp
                    bucket["FN"] += fn
                    bucket["SUM_IOU"] += sum_iou
                    bucket["GT"] += len(gt_boxes)
                log_rows.append({
                    "img_path": task["img_path_rel"],
                    "num_raw": len(detections),
                    "num_kept": len(kept),
                    "num_gt": len(gt_boxes),
                })

    # ------------------------------------------------------------------
    # report
    # ------------------------------------------------------------------
    def summarize(b):
        p = b["TP"] / (b["TP"] + b["FP"]) if b["TP"] + b["FP"] else 0.0
        r = b["TP"] / (b["TP"] + b["FN"]) if b["TP"] + b["FN"] else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        miou = b["SUM_IOU"] / b["GT"] if b["GT"] else 0.0
        return p, r, f1, miou

    lines = ["=" * 78, "Box-level localization (anomalous test images only)", "=" * 78,
             f"{'category':<14}{'P':>8}{'R':>8}{'F1':>8}{'mIoU':>8}{'GT':>7}"]
    for cat, b in stats.items():
        p, r, f1, miou = summarize(b)
        lines.append(f"{cat:<14}{p*100:>7.2f}{r*100:>7.2f}{f1*100:>7.2f}{miou*100:>7.2f}{b['GT']:>7}")
    p, r, f1, miou = summarize(global_stats)
    lines.append("-" * 78)
    lines.append(f"{'ALL':<14}{p*100:>7.2f}{r*100:>7.2f}{f1*100:>7.2f}{miou*100:>7.2f}{global_stats['GT']:>7}")

    report = "\n".join(lines)
    print("\n" + report)

    with open(os.path.join(args.output_dir, "box_level_summary.json"), "w", encoding="utf-8") as f:
        json.dump({
            "per_category": {c: {**v, "precision": summarize(v)[0],
                                 "recall": summarize(v)[1],
                                 "f1": summarize(v)[2],
                                 "mean_iou": summarize(v)[3]} for c, v in stats.items()},
            "overall": {"precision": p, "recall": r, "f1": f1, "mean_iou": miou,
                        **global_stats},
            "iou_threshold": args.iou_threshold,
            "checkpoint": args.checkpoint,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }, f, indent=2, ensure_ascii=False)

    print(f"\nBox predictions written to: {args.output_dir}")


if __name__ == "__main__":
    main()
