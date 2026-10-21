# -*- coding: utf-8 -*-
"""Qwen3-VL based anomaly localizer.

The vision-language model is asked to localize anomalous regions and to name
them with the defect vocabulary of the category (see
``configs/mvtec/category_templates.json``). Every detection is returned as
``{"label": str, "bbox_2d": [x1, y1, x2, y2], "confidence": float}`` where the
box coordinates are normalized to ``[0, 1000]``.

The produced boxes are used as geometric prompts for SAM 3 in stage 2, and the
predicted label is mapped to a hierarchical concept description from the
concept bank.
"""

import json
import os
import re

import torch
from PIL import Image, ImageColor, ImageDraw, ImageFont
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor
from vllm import LLM, SamplingParams

# vLLM spawns worker processes; keep the method explicit for reproducibility.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")


class QwenVLInferencer:
    """Thin wrapper around a Qwen3-VL model served by vLLM."""

    def __init__(
        self,
        checkpoint_path="Qwen/Qwen3-VL-2B-Instruct",
        max_model_len=8192,
        seed=0,
        tensor_parallel_size=None,
        font_path="NotoSansCJK-Regular.ttc",
        font_size=14,
    ):
        self.checkpoint_path = checkpoint_path
        self.font_path = font_path
        self.font_size = font_size

        print(f"Loading processor: {self.checkpoint_path} ...")
        self.processor = AutoProcessor.from_pretrained(self.checkpoint_path)

        print(f"Initializing vLLM engine: {self.checkpoint_path} ...")
        if tensor_parallel_size is None:
            tensor_parallel_size = torch.cuda.device_count()
        self.llm = LLM(
            model=self.checkpoint_path,
            mm_encoder_tp_mode="data",
            tensor_parallel_size=tensor_parallel_size,
            seed=seed,
            max_model_len=max_model_len,
        )
        print("Model ready.")

        self.colors = [
            "red", "green", "blue", "yellow", "orange", "pink", "purple", "brown",
            "gray", "beige", "turquoise", "cyan", "magenta", "lime", "navy",
            "maroon", "teal", "olive", "coral", "lavender", "violet", "gold", "silver",
        ] + [name for (name, _) in ImageColor.colormap.items()]

        try:
            self.font = ImageFont.truetype(self.font_path, size=self.font_size)
        except IOError:
            print(f"Warning: font '{self.font_path}' not found, falling back to default.")
            self.font = ImageFont.load_default()

        self.category_templates = {}

    # ------------------------------------------------------------------
    # inference
    # ------------------------------------------------------------------
    def prepare_inputs(self, messages):
        try:
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,  # avoid spending tokens on reasoning traces
            )
        except TypeError:
            # Older chat templates do not accept enable_thinking.
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages,
            image_patch_size=self.processor.image_processor.patch_size,
            return_video_kwargs=True,
            return_video_metadata=True,
        )
        mm_data = {}
        if image_inputs is not None:
            mm_data["image"] = image_inputs
        if video_inputs is not None:
            mm_data["video"] = video_inputs

        return {"prompt": text, "multi_modal_data": mm_data, "mm_processor_kwargs": video_kwargs}

    def generate(self, messages_list, temperature=0.0, max_tokens=1024, top_k=-1, top_p=1.0):
        inputs = [self.prepare_inputs(msg) for msg in messages_list]
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            top_k=top_k,
            top_p=top_p,
        )
        outputs = self.llm.generate(inputs, sampling_params=sampling_params)
        return [output.outputs[0].text for output in outputs]

    # ------------------------------------------------------------------
    # prompt construction
    # ------------------------------------------------------------------
    def load_category_templates(self, templates_path):
        """Load the per-category prompt templates (system prompt + user prompt)."""
        if not os.path.exists(templates_path):
            print(f"Warning: category template file not found: {templates_path}")
            return
        with open(templates_path, "r", encoding="utf-8") as f:
            templates = json.load(f)
        for category, template in templates.items():
            if isinstance(template, dict) and (
                "system_prompt" in template or "user_prompt" in template
            ):
                self.category_templates[category] = template
        print(f"Loaded {len(self.category_templates)} category templates.")

    def build_messages(self, image_input, category=None, defect_types=None):
        """Build the chat messages for a single image."""
        template = self.category_templates.get(category) if category else None
        if template is None:
            return [{
                "role": "user",
                "content": [
                    {"type": "image", "image": image_input},
                    {"type": "text", "text": (
                        'Locate every instance that belongs to the following categories: '
                        '"defect". Report bbox coordinates and confidence score in JSON format.'
                    )},
                ],
            }]

        messages = []
        if "system_prompt" in template:
            messages.append({"role": "system", "content": template["system_prompt"]})

        user_text = template["user_prompt"].replace("{image}", "")
        if category:
            user_text = user_text.replace("{category}", category)
        if defect_types:
            user_text = user_text.replace("{defect_list}", ", ".join(defect_types))

        messages.append({
            "role": "user",
            "content": [
                {"type": "image", "image": image_input},
                {"type": "text", "text": user_text.strip()},
            ],
        })
        return messages

    # ------------------------------------------------------------------
    # output parsing
    # ------------------------------------------------------------------
    def parse_json(self, json_output: str):
        """Parse a (possibly noisy) model output into a list of detections."""
        match = re.search(r"<json>(.*?)(?:</json>|$)", json_output, re.DOTALL)
        json_str = match.group(1).strip() if match else json_output

        start = json_str.find("[")
        if start != -1:
            body = json_str[start:]
            decoder = json.JSONDecoder()
            results = []
            idx = 0
            while idx < len(body):
                while idx < len(body) and body[idx].isspace():
                    idx += 1
                if idx >= len(body) or body[idx] == "]":
                    break
                if body[idx] != "{":
                    idx += 1
                    continue
                try:
                    obj, next_idx = decoder.raw_decode(body, idx)
                    if isinstance(obj, dict):
                        results.append(obj)
                    idx = next_idx
                except json.JSONDecodeError:
                    next_brace = body.find("{", idx + 1)
                    if next_brace == -1:
                        break
                    idx = next_brace
            if results:
                return self._coerce(results)

        try:
            obj = json.loads(json_str)
        except json.JSONDecodeError:
            return []

        if isinstance(obj, dict):
            if "bbox_2d" in obj:
                return self._coerce([obj])
            for key in ("bbox", "boxes", "bounding_boxes"):
                if key in obj and isinstance(obj[key], list) and len(obj[key]) == 4:
                    return self._coerce([{
                        "bbox_2d": obj[key],
                        "label": obj.get("label", "defect"),
                        "confidence": obj.get("confidence", 1.0),
                    }])
            return []
        if isinstance(obj, list):
            return self._coerce([item for item in obj if isinstance(item, dict)])
        return []

    @staticmethod
    def _coerce(items):
        for item in items:
            conf = item.get("confidence")
            if isinstance(conf, str):
                try:
                    item["confidence"] = float(conf)
                except (TypeError, ValueError):
                    item["confidence"] = 1.0
            bbox = item.get("bbox_2d")
            if isinstance(bbox, list):
                try:
                    item["bbox_2d"] = [float(v) for v in bbox]
                except (TypeError, ValueError):
                    pass
        return items

    # ------------------------------------------------------------------
    # visualization
    # ------------------------------------------------------------------
    def draw_bounding_boxes(self, image_path: str, defects: list, save_path: str = None):
        img = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(img)
        width, height = img.size

        for i, det in enumerate(defects):
            if "bbox_2d" not in det:
                continue
            color = self.colors[i % len(self.colors)]
            bbox = det["bbox_2d"]
            x1 = int(bbox[0] / 1000 * width)
            y1 = int(bbox[1] / 1000 * height)
            x2 = int(bbox[2] / 1000 * width)
            y2 = int(bbox[3] / 1000 * height)
            if x1 > x2:
                x1, x2 = x2, x1
            if y1 > y2:
                y1, y2 = y2, y1

            draw.rectangle(((x1, y1), (x2, y2)), outline=color, width=3)
            label = f"{det.get('label', 'defect')} {det.get('confidence', 1.0):.2f}"
            draw.text((x1 + 3, max(0, y1 - 16)), label, fill=color, font=self.font)

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            img.save(save_path)
        return img
