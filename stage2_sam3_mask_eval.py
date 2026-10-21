# -*- coding: utf-8 -*-
"""Stage 2: SAM 3 segmentation and box/mask P/R/F1/mIoU.

Only the anomalous test images are processed (they are the only ones with a GT
mask). The SAM 3 instance masks are binarized at 0.5 and merged into one
prediction mask per image, which is then compared with the GT mask. Mask
P/R/F1/IoU are computed per image and averaged over images, while the box
P/R/F1 are the pooled TP/FP/FN counts with greedy IoU-0.5 matching.

Pipeline per image::

    VLM boxes  --label-->  concept bank  --> SAM 3 (text + box prompt)
                                              |
                                              +--> instance masks  -> OR -> pred mask
                                              +--> instance boxes  -> score/area filter -> box metrics

Usage:
    python stage2_sam3_mask_eval.py \
        --meta_json    /path/to/mvtec/meta_mask_box.json \
        --dataset_root /path/to/mvtec \
        --vlm_pred_dir outputs/mvtec/vlm_predictions \
        --concept_bank configs/mvtec/concept_bank.json \
        --output_dir   outputs/mvtec/sam3_mask \
        --sam3_checkpoint /path/to/sam3.pt
"""

import argparse
import json
import os
from datetime import datetime

import numpy as np
from PIL import Image
from tqdm import tqdm

from method.sam3_promptable import SAM3PromptableSegmenter


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
def mask_metrics(pred_mask, gt_mask):
    """Per-image binary mask metrics: IoU, precision, recall, F1."""
    pred = pred_mask > 0
    gt = gt_mask > 0
    tp = float(np.logical_and(pred, gt).sum())
    fp = float(np.logical_and(pred, ~gt).sum())
    fn = float(np.logical_and(~pred, gt).sum())
    union = float(np.logical_or(pred, gt).sum())

    iou = tp / union if union > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return iou, precision, recall, f1


def box_iou(a, b):
    xA, yA = max(a[0], b[0]), max(a[1], b[1])
    xB, yB = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, xB - xA) * max(0.0, yB - yA)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def match_boxes(pred_boxes, gt_boxes, iou_threshold):
    """Greedy one-to-one matching; returns (tp, fp, fn, sum_iou)."""
    tp, fp, sum_iou = 0, 0, 0.0
    matched = set()
    for p in pred_boxes:
        best_iou, best_idx = 0.0, -1
        for i, g in enumerate(gt_boxes):
            if i in matched:
                continue
            iou = box_iou(p, g)
            if iou > best_iou:
                best_iou, best_idx = iou, i
        if best_idx >= 0 and best_iou >= iou_threshold:
            tp += 1
            matched.add(best_idx)
            sum_iou += best_iou
        else:
            fp += 1
    return tp, fp, len(gt_boxes) - len(matched), sum_iou


def prf1_miou(tp, fp, fn, sum_iou, gt_count):
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    miou = sum_iou / gt_count if gt_count > 0 else 0.0
    return p, r, f1, miou


# ---------------------------------------------------------------------------
# evaluator
# ---------------------------------------------------------------------------
class SAM3MaskEvaluator:
    def __init__(
        self,
        meta_json_path,
        dataset_root,
        vlm_pred_dir,
        concept_bank_path="",
        output_dir="./outputs/mvtec_sam3_mask",
        confidence_threshold=0.7,
        iou_threshold=0.5,
        min_area_ratio=0.0005,
        max_area_ratio=0.6,
        sam3_checkpoint=None,
        save_pred_mask=False,
        max_images=None,
    ):
        self.meta_json_path = meta_json_path
        self.dataset_root = dataset_root
        self.vlm_pred_dir = vlm_pred_dir
        self.output_dir = output_dir
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.min_area_ratio = min_area_ratio
        self.max_area_ratio = max_area_ratio
        self.save_pred_mask = save_pred_mask
        self.max_images = max_images

        os.makedirs(self.output_dir, exist_ok=True)

        self.concept_bank = {}
        if concept_bank_path and os.path.exists(concept_bank_path):
            with open(concept_bank_path, "r", encoding="utf-8") as f:
                self.concept_bank = json.load(f)
            print(f"Loaded concept bank: {concept_bank_path}")
        else:
            print(f"Concept bank not found ({concept_bank_path}); using the text prompt 'defect'.")

        self.sam3 = SAM3PromptableSegmenter(
            checkpoint_path=sam3_checkpoint,
            confidence_threshold=confidence_threshold,
        )

    # ---------------- data ----------------
    def _load_tasks(self):
        with open(self.meta_json_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        tasks = []
        for split, categories in meta.items():
            if split != "test":
                continue
            for category, items in categories.items():
                for item in items:
                    if item.get("anomaly") != 1:
                        continue
                    if not item.get("mask_path") or not item.get("box_json_path"):
                        continue
                    tasks.append({
                        "category": category,
                        "img_path_rel": item["img_path"],
                        "img_path_abs": os.path.join(self.dataset_root, item["img_path"]),
                        "mask_path_abs": os.path.join(self.dataset_root, item["mask_path"]),
                        "box_json_path_abs": os.path.join(self.dataset_root, item["box_json_path"]),
                    })
        return tasks

    def get_sam3_prompt(self, label, category):
        """Map a VLM label to the matching concept-bank description."""
        bank = self.concept_bank.get(category, {})
        for key in bank:
            if key != "defect" and key.lower() in label.lower():
                return bank[key]
        if "defect" in label.lower() or "anomaly" in label.lower():
            return bank.get("defect", "defect")
        return label or "defect"

    def _vlm_boxes(self, vlm_json_path, img_w, img_h, category):
        """Read stage-1 boxes -> SAM 3 prompts (normalized) + absolute boxes."""
        norm_boxes, abs_boxes, concepts = [], [], []
        if not os.path.exists(vlm_json_path):
            return norm_boxes, abs_boxes, concepts
        with open(vlm_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        predictions = data if isinstance(data, list) else data.get("predictions", [])
        if not isinstance(predictions, list):
            predictions = [predictions]

        for item in predictions:
            bbox = item.get("bbox_2d")
            if not bbox or len(bbox) != 4:
                continue
            x1, x2 = sorted((bbox[0] / 1000.0 * img_w, bbox[2] / 1000.0 * img_w))
            y1, y2 = sorted((bbox[1] / 1000.0 * img_h, bbox[3] / 1000.0 * img_h))
            norm_boxes.append([(x1 + x2) / 2.0 / img_w, (y1 + y2) / 2.0 / img_h,
                               (x2 - x1) / img_w, (y2 - y1) / img_h])
            abs_boxes.append([x1, y1, x2, y2])
            concepts.append(self.get_sam3_prompt(item.get("label", ""), category))
        return norm_boxes, abs_boxes, concepts

    @staticmethod
    def _gt_boxes(box_json_path):
        with open(box_json_path, "r", encoding="utf-8") as f:
            return json.load(f).get("gt_boxes", [])

    # ---------------- main ----------------
    def run(self):
        tasks = self._load_tasks()
        if self.max_images:
            tasks = tasks[: self.max_images]
        print(f"Loaded {len(tasks)} anomalous test images.")

        cat_mask = {}
        cat_box = {}
        g_mask = {"iou": 0.0, "p": 0.0, "r": 0.0, "f1": 0.0, "n": 0}
        g_box = {"TP": 0, "FP": 0, "FN": 0, "SUM_IOU": 0.0, "GT": 0}

        for task in tqdm(tasks, desc="stage2-mask: SAM 3"):
            cat = task["category"]
            cat_mask.setdefault(cat, {"iou": 0.0, "p": 0.0, "r": 0.0, "f1": 0.0, "n": 0})
            cat_box.setdefault(cat, {"TP": 0, "FP": 0, "FN": 0, "SUM_IOU": 0.0, "GT": 0})

            img_path = task["img_path_abs"]
            with Image.open(img_path) as im:
                img_w, img_h = im.size
            gt_mask = np.array(Image.open(task["mask_path_abs"]).convert("L")) > 0
            gt_boxes = self._gt_boxes(task["box_json_path_abs"])

            rel_dir = os.path.dirname(task["img_path_rel"])
            stem = os.path.splitext(os.path.basename(task["img_path_rel"]))[0]
            vlm_json = os.path.join(self.vlm_pred_dir, rel_dir, f"{stem}_qwen.json")

            norm_boxes, _, concepts = self._vlm_boxes(vlm_json, img_w, img_h, cat)

            pred_mask = np.zeros((img_h, img_w), dtype=bool)
            sam_boxes, sam_scores = [], []
            for box, concept in zip(norm_boxes, concepts):
                out = self.sam3.segment(
                    image_path=img_path, text_prompt=concept, boxes_cxcywh_norm=[box]
                )
                probs = out.get("pred_masks")
                if probs is not None:
                    for pm in probs:
                        pred_mask |= (pm > 0)
                for b, s in zip(out.get("pred_boxes_absolute") or [],
                                out.get("pred_scores") or []):
                    sam_boxes.append(b)
                    sam_scores.append(s)

            # instance-level box filtering (confidence + plausible area)
            img_area = float(img_w * img_h)
            final_boxes = []
            for box, score in zip(sam_boxes, sam_scores):
                if score < self.confidence_threshold:
                    continue
                area = (box[2] - box[0]) * (box[3] - box[1])
                if area < self.min_area_ratio * img_area or area > self.max_area_ratio * img_area:
                    continue
                final_boxes.append(box)

            tp, fp, fn, sum_iou = match_boxes(final_boxes, gt_boxes, self.iou_threshold)
            for bucket in (cat_box[cat], g_box):
                bucket["TP"] += tp
                bucket["FP"] += fp
                bucket["FN"] += fn
                bucket["SUM_IOU"] += sum_iou
                bucket["GT"] += len(gt_boxes)

            iou, p, r, f1 = mask_metrics(pred_mask, gt_mask)
            for bucket in (cat_mask[cat], g_mask):
                bucket["iou"] += iou
                bucket["p"] += p
                bucket["r"] += r
                bucket["f1"] += f1
                bucket["n"] += 1

            if self.save_pred_mask:
                out_dir = os.path.join(self.output_dir, "pred_masks", rel_dir)
                os.makedirs(out_dir, exist_ok=True)
                Image.fromarray((pred_mask * 255).astype(np.uint8)).save(
                    os.path.join(out_dir, f"{stem}_sam_mask.png")
                )

        results = self._summarize(cat_mask, cat_box, g_mask, g_box)
        self._report(results)
        summary_path = os.path.join(self.output_dir, "mask_box_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nSummary written to {summary_path}")
        return results

    @staticmethod
    def _summarize(cat_mask, cat_box, g_mask, g_box):
        def mask_avg(m):
            n = m["n"] or 1
            return {k: m[k] / n for k in ("iou", "p", "r", "f1")}

        per_cat = {}
        for cat in cat_mask:
            cb = cat_box[cat]
            p, r, f1, miou = prf1_miou(cb["TP"], cb["FP"], cb["FN"], cb["SUM_IOU"], cb["GT"])
            per_cat[cat] = {
                "num_images": cat_mask[cat]["n"],
                "box": {"precision": p, "recall": r, "f1": f1, "miou": miou},
                "mask": mask_avg(cat_mask[cat]),
            }

        p, r, f1, miou = prf1_miou(g_box["TP"], g_box["FP"], g_box["FN"],
                                   g_box["SUM_IOU"], g_box["GT"])
        return {
            "box_level": {"precision": p, "recall": r, "f1": f1, "miou": miou,
                          **g_box, "num_images": g_mask["n"]},
            "mask_level": mask_avg(g_mask),
            "per_category": per_cat,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }

    def _report(self, results):
        b, m = results["box_level"], results["mask_level"]
        lines = [
            "=" * 84,
            f"{'category':<14}{'box P':>8}{'box R':>8}{'box F1':>8}{'box mIoU':>10}"
            f"{'mask P':>9}{'mask R':>8}{'mask F1':>9}{'mask mIoU':>10}",
            "=" * 84,
        ]
        for cat, v in results["per_category"].items():
            cb, cm = v["box"], v["mask"]
            lines.append(
                f"{cat:<14}{cb['precision']*100:>7.2f}{cb['recall']*100:>8.2f}"
                f"{cb['f1']*100:>8.2f}{cb['miou']*100:>10.2f}"
                f"{cm['p']*100:>9.2f}{cm['r']*100:>8.2f}{cm['f1']*100:>9.2f}{cm['iou']*100:>10.2f}"
            )
        lines.append("-" * 84)
        lines.append(
            f"{'ALL':<14}{b['precision']*100:>7.2f}{b['recall']*100:>8.2f}"
            f"{b['f1']*100:>8.2f}{b['miou']*100:>10.2f}"
            f"{m['p']*100:>9.2f}{m['r']*100:>8.2f}{m['f1']*100:>9.2f}{m['iou']*100:>10.2f}"
        )
        print("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="Stage 2 (mask-level): SAM 3 + box/mask P/R/F1/mIoU")
    parser.add_argument("--meta_json", required=True)
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--vlm_pred_dir", required=True)
    parser.add_argument("--concept_bank", default="configs/mvtec/concept_bank.json")
    parser.add_argument("--output_dir", default="./outputs/mvtec_sam3_mask")
    parser.add_argument("--confidence_threshold", type=float, default=0.7,
                        help="SAM 3 instance confidence threshold")
    parser.add_argument("--iou_threshold", type=float, default=0.5)
    parser.add_argument("--min_area_ratio", type=float, default=0.0005)
    parser.add_argument("--max_area_ratio", type=float, default=0.6)
    parser.add_argument("--sam3_checkpoint", default=None)
    parser.add_argument("--save_pred_mask", action="store_true")
    parser.add_argument("--max_images", type=int, default=None)
    args = parser.parse_args()

    evaluator = SAM3MaskEvaluator(
        meta_json_path=args.meta_json,
        dataset_root=args.dataset_root,
        vlm_pred_dir=args.vlm_pred_dir,
        concept_bank_path=args.concept_bank,
        output_dir=args.output_dir,
        confidence_threshold=args.confidence_threshold,
        iou_threshold=args.iou_threshold,
        min_area_ratio=args.min_area_ratio,
        max_area_ratio=args.max_area_ratio,
        sam3_checkpoint=args.sam3_checkpoint,
        save_pred_mask=args.save_pred_mask,
        max_images=args.max_images,
    )
    evaluator.run()


if __name__ == "__main__":
    main()
