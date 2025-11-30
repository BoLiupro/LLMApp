#!/usr/bin/env bash
# scripts/run_predictor.sh
# 用法：
#   chmod +x scripts/run_predictor.sh
#   scripts/run_predictor.sh
# run.sh 顶部环境变量区加一行
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1



set -Eeuo pipefail

# 运行入口（按你的项目结构来改）
PYTHON=${PYTHON:-python}
ENTRY="run.py"      
# ENTRY="test.py"     
TRAIN="True"

ABLATION="False"          # 是否使用消融版本的 PredictorTrainer
SELECTOR_CKPT="checkpt/nanchang/bestSelector_12.pt"
PREDICTOR_CKPT="checkpt/nanchang/bestPredictor.pt"
CONTINUE_THE_BEST="checkpt/nanchang/bestPredictor.pt"      # 如果要接着上次训练继续，就填上次的 bestPredictor.pt 路径；否则留空

# =========================
# 重要参数区（按需修改）
# =========================

# 数据源（支持目录 / 文件 / 通配符；可多个）
CSV_FILES=(
  "data/processed_data/nanchang/app_usage_records"
  # "data/processed_data/nanchang/app_usage_records/app_usage_record_20160421.csv"
)
DEVICE="cuda"                 # cuda / cpu
SELECTOR_BATCH_SIZE=128
PREDICTOR_BATCH_SIZE=32
SELECTOR_EPOCHS=10
PREDICTOR_EPOCHS=5
LONG_SEQ_LEN=64
SHORT_SEQ_LEN=12
KEY_PATTERN_LEN=12
LR=1e-4
WEIGHT_DECAY=0.01
LOG_EVERY=50                 # 多少步打印一次训练日志

# ------- vocab/size -------
APP_VOCAB_SIZE=640
LOCATION_VOCAB_SIZE=63200
NUM_APP_CATE=23
NUM_TRAFFIC_BIN=10

# ------- resource paths -------
APP_CATEGORY_CSV="data/processed_data/nanchang/category.csv"
LOCATION_CSV="data/processed_data/nanchang/location.csv"
TRAFFIC_BIN_LABELS="ultra-low,very-low,low,lower-mid,mid,upper-mid,high,very-high,extreme,burst"

# （可选）模型与训练其他开关
MODEL_PATH="/datadisk/gemma-2-2b"
FREEZE_BACKBONE=true          # 你的 config 用 str2bool；这里写 true/false 即可
NUM_WORKERS=4
PIN_MEMORY=true
VAL_RATIO=0.05
PAD_ID=0
STRIDE=""                     # 留空相当于 None；要设置就写整数，比如 32
SELECTOR_TOPK=5
CATE_AUX_LAMBDA=2
TRAF_AUX_LAMBDA=0.8
EVAL_TOPK=(1 5 10)

# =========================
# 日志与环境（可选）
# =========================
TS=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/run_${TS}.log"

echo "==> Logging to $LOG_FILE"

# =========================
# 组装命令
# =========================

# 把 CSV 数组展开成多个 --csv_file 参数（nargs='+')
CSV_ARGS=(--csv_file "${CSV_FILES[@]}")

EVAL_ARGS=(--eval_topk "${EVAL_TOPK[@]}")

# 如果 STRIDE 为空，则不传参；否则传 --stride 值
STRIDE_ARGS=()
if [[ -n "${STRIDE}" ]]; then
  STRIDE_ARGS=(--stride "${STRIDE}")
fi

CMD=(
  "${PYTHON}" -u "${ENTRY}"          # ← 这里加上 -u
  # "${PYTHON}" "${ENTRY}"
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
  --predictor_save_path "${PREDICTOR_CKPT}"
  --continue_the_best "${CONTINUE_THE_BEST}"
)

# 打印并执行
echo "==> Command:"
printf ' %q' "${CMD[@]}"; echo
echo "--------------------------------------------"

# 记录日志到文件，同时在终端显示
"${CMD[@]}" 2>&1 | tee -a "${LOG_FILE}"
