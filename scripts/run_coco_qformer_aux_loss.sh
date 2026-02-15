#!/usr/bin/env bash
set -euo pipefail

# COCO Q-Former training with auxiliary loss to prevent modality collapse
# This script implements the reconstruction loss solution to fix the collapse issue

MODEL_PATH="${MODEL_PATH:-/data/ssd_storage/llama/}"
VISION_TOWER="${VISION_TOWER:-openai/clip-vit-large-patch14-336}"
DATA_PATH="${DATA_PATH:-/home/kezouke/Llava/coco/coco_train2014_llava.json}"
IMAGE_FOLDER="${IMAGE_FOLDER:-/home/kezouke/Llava/coco/train2014}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./checkpoints}"

# Dataset tag for naming outputs (defaults to last folder name of IMAGE_FOLDER)
DATASET_TAG="${DATASET_TAG:-$(basename "$IMAGE_FOLDER")}"

# Deepspeed (optional)
USE_DEEPSPEED="${USE_DEEPSPEED:-false}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-LLaVA-NeXT/scripts/zero2.json}"

# Precision
BF16="${BF16:-true}"
FP16="${FP16:-false}"

# Training hyperparameters
SEED="${SEED:-42}"
EPOCHS="${EPOCHS:-3}"
PER_DEVICE_BATCH="${PER_DEVICE_BATCH:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-16}"
LR="${LR:-2e-4}"
WD="${WD:-0.01}"
WARMUP="${WARMUP:-0.03}"
SCHED="${SCHED:-cosine}"
MAXLEN="${MAXLEN:-4096}"
LOG_STEPS="${LOG_STEPS:-10}"
SAVE_STEPS="${SAVE_STEPS:-1000}"
SAVE_TOTAL="${SAVE_TOTAL:-2}"
TF32="${TF32:-true}"
NUM_WORKERS="${NUM_WORKERS:-4}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

# Q-Former specific hyperparameters
QFORMER_DEPTH="${QFORMER_DEPTH:-3}"
QFORMER_LATENTS="${QFORMER_LATENTS:-32}"
QFORMER_CROSS_ATTENTION_FREQ="${QFORMER_CROSS_ATTENTION_FREQ:-1}"

# Auxiliary loss configuration
AUX_LOSS_WEIGHT="${AUX_LOSS_WEIGHT:-0.1}"
QFORMER_TUNE_STRATEGY="${QFORMER_TUNE_STRATEGY:-cross_attn_only}"

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

# Output directory
OUTDIR="${OUTPUT_ROOT}/llava-qformer-aux-loss-d${QFORMER_DEPTH}-l${QFORMER_LATENTS}-${DATASET_TAG}"
mkdir -p "$OUTDIR"
echo "Launching Q-Former training with auxiliary loss -> ${OUTDIR}"

# Build command
CMD=(python -m llava.train.train
  --model_name_or_path "$MODEL_PATH"
  --version "llama_v3"
  --vision_tower "$VISION_TOWER"
  --data_path "$DATA_PATH"
  --image_folder "$IMAGE_FOLDER"
  --mm_resampler_type "qformer"
  --mm_qformer_depth "$QFORMER_DEPTH"
  --mm_qformer_latents "$QFORMER_LATENTS"
  --mm_qformer_cross_attention_freq "$QFORMER_CROSS_ATTENTION_FREQ"
  --mm_qformer_use_aux_loss True
  --mm_qformer_aux_loss_weight "$AUX_LOSS_WEIGHT"
  --mm_tune_qformer_layers "$QFORMER_TUNE_STRATEGY"
  --mm_tunable_parts "mm_vision_resampler,mm_mlp_adapter"
  --mm_projector_type "linear"
  --mm_patch_merge_type flat
  --image_aspect_ratio square
  --mm_vision_select_layer -1
  --freeze_backbone True
  --tune_mm_mlp_adapter True
  --tune_mm_vision_resampler True
  --num_train_epochs "$EPOCHS"
  --per_device_train_batch_size "$PER_DEVICE_BATCH"
  --gradient_accumulation_steps "$GRAD_ACCUM"
  --learning_rate "$LR"
  --weight_decay "$WD"
  --warmup_ratio "$WARMUP"
  --lr_scheduler_type "$SCHED"
  --seed "$SEED"
  --attn_implementation sdpa
  --remove_unused_columns False
  --model_max_length "$MAXLEN"
  --logging_steps "$LOG_STEPS"
  --save_steps "$SAVE_STEPS"
  --save_total_limit "$SAVE_TOTAL"
  --gradient_checkpointing True
  --output_dir "$OUTDIR"
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
CMD+=(--evaluation_strategy no --save_strategy steps --report_to none --dataloader_num_workers "$NUM_WORKERS" --lazy_preprocess True --overwrite_output_dir True)

# TF32 toggle
if [[ "$TF32" == "true" ]]; then
  CMD+=(--tf32 True)
fi

# User extras
if [[ -n "$EXTRA_ARGS" ]]; then
  # shellcheck disable=SC2206
  EXTRA_ARR=($EXTRA_ARGS)
  CMD+=("${EXTRA_ARR[@]}")
fi

# Print and run
printf 'Command: %q ' "${CMD[@]}" | tee "$OUTDIR/command.txt"
echo
"${CMD[@]}" 2>&1 | tee "$OUTDIR/train.log"

echo "Q-Former training with auxiliary loss finished. Output under ${OUTDIR}"
echo "To test for modality collapse, run: python prove_collapse.py --checkpoint ${OUTDIR}/checkpoint-<N>"