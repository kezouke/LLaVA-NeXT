#!/usr/bin/env bash
set -euo pipefail

# COCO projector ablations launcher: A1 (linear), A6 (mlp2x_gelu), A7 (mlp2x_res2x_gelu), A8 (pooler)
# Backbones frozen, projector-only training. Matches early fusion injection in llava and projector selection in code.

MODEL_PATH="${MODEL_PATH:-/data/ssd_storage/llama/}"
VISION_TOWER="${VISION_TOWER:-openai/clip-vit-large-patch14-336}"
DATA_PATH="${DATA_PATH:-/home/kezouke/Llava/coco/coco_train2014_llava.json}"
IMAGE_FOLDER="${IMAGE_FOLDER:-/home/kezouke/Llava/coco/train2014}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./checkpoints}"

# Dataset tag for naming outputs (defaults to last folder name of IMAGE_FOLDER)
DATASET_TAG="${DATASET_TAG:-$(basename "$IMAGE_FOLDER")}"

# Deepspeed (optional)
USE_DEEPSPEED="${USE_DEEPSPEED:-false}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-LLaVA-NeXT/scripts/zero2_torch_adamw.json}"

# Precision
BF16="${BF16:-true}"
FP16="${FP16:-false}"

# Training hyperparameters (shared)
SEED="${SEED:-42}"
EPOCHS="${EPOCHS:-3}"
PER_DEVICE_BATCH="${PER_DEVICE_BATCH:-8}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
LR="${LR:-2e-4}"
WD="${WD:-0.01}"
WARMUP="${WARMUP:-0.03}"
SCHED="${SCHED:-cosine}"
MAXLEN="${MAXLEN:-4096}"
LOG_STEPS="${LOG_STEPS:-10}"
SAVE_STEPS="${SAVE_STEPS:-1000}"
SAVE_TOTAL="${SAVE_TOTAL:-2}"
PROJ_LR="${PROJ_LR:-}"
TF32="${TF32:-true}"
NUM_WORKERS="${NUM_WORKERS:-16}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

# Sanity checks
if [[ ! -d "$MODEL_PATH" && "$MODEL_PATH" != *"/"* ]]; then
  echo "Warning: MODEL_PATH '$MODEL_PATH' is not a local dir; assuming HF hub id."
fi
if [[ ! -f "$DATA_PATH" ]]; then
  echo "Error: DATA_PATH '$DATA_PATH' not found." >&2
  exit 1
fi
if [[ ! -d "$IMAGE_FOLDER" ]]; then
  echo "Error: IMAGE_FOLDER '$IMAGE_FOLDER' not found." >&2
  exit 1
fi
if [[ "$USE_DEEPSPEED" == "true" && ! -f "$DEEPSPEED_CONFIG" ]]; then
  echo "Error: DEEPSPEED_CONFIG '$DEEPSPEED_CONFIG' not found." >&2
  exit 1
fi

run_ablation () {
  local name="$1"
  local projector="$2"
  local proj_slug="$projector"
  if [[ "$projector" == "mlp2x_gelu" ]]; then proj_slug="mlp2x"; fi
  if [[ "$projector" == "mlp2x_res2x_gelu" ]]; then proj_slug="mlp2x-res2x"; fi
  local outdir="${OUTPUT_ROOT}/llava-${proj_slug}-${DATASET_TAG}"
  mkdir -p "$outdir"
  echo "Launching ablation ${name} with projector=${projector} -> ${outdir}"

  # Build command
  CMD=(python -m llava.train.train
    --model_name_or_path "$MODEL_PATH"
    --vision_tower "$VISION_TOWER"
    --data_path "$DATA_PATH"
    --image_folder "$IMAGE_FOLDER"
    --mm_projector_type "$projector"
    --mm_patch_merge_type flat
    --image_aspect_ratio square
    --mm_vision_select_layer -1
    --freeze_backbone True
    --tune_mm_mlp_adapter True
    --tune_mm_vision_resampler False
    --num_train_epochs "$EPOCHS"
    --per_device_train_batch_size "$PER_DEVICE_BATCH"
    --gradient_accumulation_steps "$GRAD_ACCUM"
    --learning_rate "$LR"
    --weight_decay "$WD"
    --warmup_ratio "$WARMUP"
    --lr_scheduler_type "$SCHED"
    --seed "$SEED"
    --attn_implementation flash_attention_2
    --remove_unused_columns False
    --model_max_length "$MAXLEN"
    --logging_steps "$LOG_STEPS"
    --save_steps "$SAVE_STEPS"
    --save_total_limit "$SAVE_TOTAL"
    --output_dir "$outdir"
  )

  # Precision flags
  if [[ "$BF16" == "true" ]]; then
    CMD+=(--bf16 True)
  elif [[ "$FP16" == "true" ]]; then
    CMD+=(--fp16 True)
  fi

  # Deepspeed
  if [[ "$USE_DEEPSPEED" == "true" ]]; then
    CMD+=(--deepspeed "$DEEPSPEED_CONFIG")
  fi

  # Common training I/O and loader options
  CMD+=(--evaluation_strategy no --save_strategy steps --report_to none --dataloader_num_workers "$NUM_WORKERS" --lazy_preprocess True)

  # TF32 toggle
  if [[ "$TF32" == "true" ]]; then
    CMD+=(--tf32 True)
  fi

  # Projector-specific LR (optional)
  if [[ -n "$PROJ_LR" ]]; then
    CMD+=(--mm_projector_lr "$PROJ_LR")
  fi

  # User extras
  if [[ -n "$EXTRA_ARGS" ]]; then
    # shellcheck disable=SC2206
    EXTRA_ARR=($EXTRA_ARGS)
    CMD+=("${EXTRA_ARR[@]}")
  fi

  # Print and run
  printf 'Command: %q ' "${CMD[@]}" | tee "$outdir/command.txt"
  echo
  "${CMD[@]}" 2>&1 | tee "$outdir/train.log"
}

# A1: linear
run_ablation "A1_linear" "linear"

# A6: mlp2x_gelu
run_ablation "A6_mlp2x" "mlp2x_gelu"

# A7: mlp2x_res2x_gelu
run_ablation "A7_mlp2x_res2x" "mlp2x_res2x_gelu"

# A8: pooler
run_ablation "A8_pooler" "pooler"

echo "All ablations finished. Outputs under ${OUTPUT_ROOT}_*"