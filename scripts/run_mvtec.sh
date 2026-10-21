#!/usr/bin/env bash
# End-to-end HSCKAD run on MVTec-AD (15 categories).
#
# Stage 1 needs vLLM (Qwen3-VL); stage 2 needs SAM 3. They can live in the same
# environment or in two different ones -- if you keep them separate, run the two
# stages with their own interpreter (PY1 / PY2 below).
set -e

# ----------------------------- configuration -------------------------------
MVTEC_ROOT=${MVTEC_ROOT:-/path/to/mvtec_anomaly_detection}
VLM_CKPT=${VLM_CKPT:-Qwen/Qwen3-VL-2B-Instruct}
SAM3_CKPT=${SAM3_CKPT:-}                 # empty -> download from HF facebook/sam3
OUT=${OUT:-./outputs/mvtec}

PY1=${PY1:-python}                        # interpreter with vLLM, for stage 1
PY2=${PY2:-python}                        # interpreter with SAM 3, for stage 2

META=${MVTEC_ROOT}/meta_mask_box.json
CONFIG_DIR=configs/mvtec
VLM_OUT=${OUT}/vlm_predictions
SAM3_MASK_OUT=${OUT}/sam3_mask

# ----------------------------- 0. metadata --------------------------------
if [ ! -f "${META}" ]; then
  echo "[0/3] building ${META}"
  python tools/prepare_mvtec.py --dataset_root "${MVTEC_ROOT}"
fi

# ------------------- 1. VLM localization (boxes + PC) ---------------------
echo "[1/3] stage 1: VLM localization"
${PY1} stage1_vlm_predict.py \
  --meta_json         "${META}" \
  --dataset_root      "${MVTEC_ROOT}" \
  --checkpoint        "${VLM_CKPT}" \
  --category_templates "${CONFIG_DIR}/category_templates.json" \
  --category_config    "${CONFIG_DIR}/category_config.json" \
  --defect_types       "${CONFIG_DIR}/defect_types.json" \
  --output_dir        "${VLM_OUT}" \
  --batch_size 1

# ------------- 2. SAM 3 segmentation + box/mask P/R/F1/mIoU ----------------
SAM3_ARGS=()
if [ -n "${SAM3_CKPT}" ]; then
  SAM3_ARGS+=(--sam3_checkpoint "${SAM3_CKPT}")
fi

echo "[2/3] stage 2: SAM 3 + box/mask metrics"
${PY2} stage2_sam3_mask_eval.py \
  --meta_json      "${META}" \
  --dataset_root   "${MVTEC_ROOT}" \
  --vlm_pred_dir   "${VLM_OUT}" \
  --concept_bank   "${CONFIG_DIR}/concept_bank.json" \
  --output_dir     "${SAM3_MASK_OUT}" \
  --confidence_threshold 0.7 \
  "${SAM3_ARGS[@]}"

echo "[3/3] done"
echo "  box-level (VLM)    : ${VLM_OUT}/box_level_summary.json"
echo "  box + mask metrics : ${SAM3_MASK_OUT}/mask_box_summary.json"
