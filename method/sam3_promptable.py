# -*- coding: utf-8 -*-
"""SAM 3 promptable segmenter.

SAM 3 accepts a text prompt and/or geometric (box) prompts. HSCKAD feeds it

* the concept description that corresponds to the label predicted by the VLM
  (hierarchical concept bank lookup), and
* the VLM box for that instance.

The returned instance masks are binarized at 0.5; stage 2 merges them into one
prediction mask per image and compares it with the ground-truth mask.
"""

import os

import cv2
from PIL import Image


class SAM3PromptableSegmenter:
    """Thin wrapper around SAM 3 image inference."""

    def __init__(self, checkpoint_path=None, confidence_threshold=0.7):
        """Args:
            checkpoint_path: path to ``sam3.pt``. ``None`` -> download from the
                gated Hugging Face repo ``facebook/sam3`` (requires ``hf auth login``).
            confidence_threshold: instance-level score threshold applied inside
                SAM 3 when it keeps the candidate instances.
        """
        import sam3
        from sam3 import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        print("Initializing SAM 3 segmenter ...")
        sam3_root = os.path.dirname(sam3.__file__)
        bpe_path = os.path.join(sam3_root, "assets", "bpe_simple_vocab_16e6.txt.gz")

        self.model = build_sam3_image_model(
            bpe_path=bpe_path, checkpoint_path=checkpoint_path
        )
        self.processor = Sam3Processor(self.model, confidence_threshold=confidence_threshold)
        print("SAM 3 ready.")

    def segment(self, image_path: str, text_prompt: str, boxes_cxcywh_norm: list):
        """Segment one image.

        Args:
            image_path: image file.
            text_prompt: concept description used as the SAM 3 text prompt.
            boxes_cxcywh_norm: list of ``[cx, cy, w, h]`` boxes normalized to
                ``[0, 1]``. Empty list -> text-only prompting.

        Returns:
            dict with
                ``pred_masks``: ``(N, H, W)`` bool masks,
                ``pred_scores``: per-instance score,
                ``pred_boxes_absolute``: ``[x0, y0, x1, y1]`` per instance,
                ``orig_img_h`` / ``orig_img_w``.
        """
        if image_path.lower().endswith(".tif"):
            cvimg = cv2.imread(image_path)
            image = Image.fromarray(cv2.cvtColor(cvimg, cv2.COLOR_BGR2RGB))
        else:
            image = Image.open(image_path).convert("RGB")
        orig_img_w, orig_img_h = image.size

        state = self.processor.set_image(image)

        if text_prompt:
            state = self.processor.set_text_prompt(prompt=text_prompt, state=state)

        for box in boxes_cxcywh_norm or []:
            state = self.processor.add_geometric_prompt(box=box, label=True, state=state)

        mask_logits = state.get("masks_logits")
        if mask_logits is not None and mask_logits.numel() > 0:
            pred_masks = (mask_logits.squeeze(1) > 0.5).cpu().numpy()
        else:
            pred_masks = None

        scores = state.get("scores")
        scores = scores.tolist() if scores is not None else None

        boxes = state.get("boxes")
        pred_boxes = boxes.cpu().numpy().tolist() if boxes is not None else None

        return {
            "orig_img_h": orig_img_h,
            "orig_img_w": orig_img_w,
            "input_text": text_prompt,
            "input_boxes_norm": boxes_cxcywh_norm,
            "pred_masks": pred_masks,
            "pred_scores": scores,
            "pred_boxes_absolute": pred_boxes,
        }
