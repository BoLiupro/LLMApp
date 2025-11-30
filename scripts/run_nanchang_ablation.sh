#!/usr/bin/env bash
# scripts/run_predictor_ablation.sh
# 用法：
#   chmod +x scripts/run_predictor_ablation.sh
#   scripts/run_predictor_ablation.sh

set -Eeuo pipefail
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

# ---------------- 基本配置（按需改） ----------------
PYTHON=${PYTHON:-python}
ENTRY="run.py"

# 是否训练 Predictor（PredictorTrainer 里有 args.train 判断）
TRAIN="True"
ABLATION="True"                          # 使用 ablation 分支（Predictor_ablation）

# 选择器 & 预测器 ckpt
SELECTOR_CKPT="checkpt/nanchang/bestSelector_12.pt"   # 已训练好的 selector
OUT_DIR="checkpt/nanchang/ablation"                   # 各变体输出目录
mkdir -p "${OUT_DIR}"

# 数据
CSV_FILES=("data/processed_data/nanchang/app_usage_records")
APP_CATEGORY_CSV="data/processed_data/nanchang/category.csv"
LOCATION_CSV="data/processed_data/nanchang/location.csv"
TRAFFIC_BIN_LABELS="ultra-low,very-low,low,lower-mid,mid,upper-mid,high,very-high,extreme,burst"

# 设备 & 训练超参
DEVICE="cuda"
SELECTOR_BATCH_SIZE=128
PREDICTOR_BATCH_SIZE=32
SELECTOR_EPOCHS=15                    # 通常不再训 selector；若 run.py 必训可保留 >0
PREDICTOR_EPOCHS=2
LR=1e-4
WEIGHT_DECAY=0.01
LOG_EVERY=50

# seq 长度
LONG_SEQ_LEN=64
SHORT_SEQ_LEN=12
KEY_PATTERN_LEN=12

# 词表规模
APP_VOCAB_SIZE=640
LOCATION_VOCAB_SIZE=63200
NUM_APP_CATE=23
NUM_TRAFFIC_BIN=10

# LLM/LoRA
MODEL_PATH="/datadisk/gemma-2-2b"
FREEZE_BACKBONE=true
NUM_WORKERS=4
PIN_MEMORY=true
VAL_RATIO=0.05
PAD_ID=0
STRIDE=""   # 为空不传

# 评估
SELECTOR_TOPK=5
CATE_AUX_LAMBDA=2
TRAF_AUX_LAMBDA=1
EVAL_TOPK=(1 5 10)

# ---------------- 共同参数组装 ----------------
CSV_ARGS=(--csv_file "${CSV_FILES[@]}")
EVAL_ARGS=(--eval_topk "${EVAL_TOPK[@]}")
STRIDE_ARGS=()
[[ -n "${STRIDE}" ]] && STRIDE_ARGS=(--stride "${STRIDE}")

BASE_CMD=(
  "${PYTHON}" -u "${ENTRY}"
  "${CSV_ARGS[@]}"
  --device "${DEVICE}"
  --selector_batch_size "${SELECTOR_BATCH_SIZE}"
  --predictor_batch_size "${PREDICTOR_BATCH_SIZE}"
  --selector_epochs "${SELECTOR_EPOCHS}"
  --predictor_epochs "${PREDICTOR_EPOCHS}"
  --long_seq_len "${LONG_SEQ_LEN}"
  --short_seq_len "${SHORT_SEQ_LEN}"
  --lr "${LR}"
  --weight_decay "${WEIGHT_DECAY}"
  --log_every "${LOG_EVERY}"
  --key_pattern_len "${KEY_PATTERN_LEN}"

  --app_vocab_size "${APP_VOCAB_SIZE}"
  --location_vocab_size "${LOCATION_VOCAB_SIZE}"
  --num_app_cate "${NUM_APP_CATE}"
  --num_traffic_bin "${NUM_TRAFFIC_BIN}"

  --app_category_csv "${APP_CATEGORY_CSV}"
  --location_csv "${LOCATION_CSV}"
  --traffic_bin_labels "${TRAFFIC_BIN_LABELS}"

  --model_path "${MODEL_PATH}"
  --freeze_backbone "${FREEZE_BACKBONE}"
  --num_workers "${NUM_WORKERS}"
  --pin_memory "${PIN_MEMORY}"
  --val_ratio "${VAL_RATIO}"
  --pad_id "${PAD_ID}"
  "${STRIDE_ARGS[@]}"

  --selector_topk "${SELECTOR_TOPK}"
  --cate_aux_lambda "${CATE_AUX_LAMBDA}"
  --traf_aux_lambda "${TRAF_AUX_LAMBDA}"
  "${EVAL_ARGS[@]}"

  --ablation "${ABLATION}"
  --train "${TRAIN}"
  --selector_save_path "${SELECTOR_CKPT}"
)

# ---------------- 三个变体配置 ----------------
declare -a NAMES=("w_o_nextcate" "w_o_hlong" "only_llm")
declare -a USE_NEXT=("false"      "true"      "false")
declare -a USE_HLONG=("true"      "false"     "false")

TS=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="logs"
mkdir -p "${LOG_DIR}"

for i in "${!NAMES[@]}"; do
  TAG="${NAMES[$i]}"
  UN="${USE_NEXT[$i]}"
  UH="${USE_HLONG[$i]}"

  PRED_PTH="${OUT_DIR}/bestPredictor_${TAG}.pt"
  LOG_FILE="${LOG_DIR}/run_ablation_${TAG}_${TS}.log"

  CMD=(
    "${BASE_CMD[@]}"
    --predictor_save_path "${PRED_PTH}"
    --use_nextcate_in_phaseB "${UN}"
    --use_hlong_in_phaseB "${UH}"
  )

  echo "=================================================="
  echo "Run ablation: ${TAG}  (use_nextcate_in_phaseB=${UN}, use_hlong_in_phaseB=${UH})"
  echo "Predictor ckpt -> ${PRED_PTH}"
  echo "Log -> ${LOG_FILE}"
  echo "CMD:"
  printf ' %q' "${CMD[@]}"; echo
  echo "--------------------------------------------------"

  "${CMD[@]}" 2>&1 | tee -a "${LOG_FILE}"
done

echo "All ablation runs finished. ✅"
