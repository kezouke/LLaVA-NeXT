#!/usr/bin/env bash
set -euo pipefail

# Stage 2: Instruction tuning for Q-Former
# Loads stage 1 projector+resampler weights, trains with llava_llama_3 template.
#
# Prerequisites:
#   1. Stage 1 completed: ./checkpoints/llava-qformer-d3-l32-train2014/mm_projector.bin exists
#   2. Instruction data created:
#      python LLaVA-NeXT/scripts/convert_caption_to_instruct.py \
#        --input /home/kezouke/Llava/coco/coco_train2014_llava.json \
#        --output /home/kezouke/Llava/coco/coco_train2014_instruct.json

MODEL_PATH="${MODEL_PATH:-/data/ssd_storage/llama/}"
VISION_TOWER="${VISION_TOWER:-openai/clip-vit-large-patch14-336}"
DATA_PATH="${DATA_PATH:-/home/kezouke/Llava/coco/coco_train2014_instruct.json}"
IMAGE_FOLDER="${IMAGE_FOLDER:-/home/kezouke/Llava/coco/train2014}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./checkpoints}"

# Stage 1 checkpoint (projector + resampler weights)
STAGE1_ADAPTER="${STAGE1_ADAPTER:-./checkpoints/llava-qformer-d3-l32-train2014/mm_projector.bin}"

# Q-Former settings (must match stage 1)
QFORMER_DEPTH="${QFORMER_DEPTH:-3}"
QFORMER_LATENTS="${QFORMER_LATENTS:-32}"
QFORMER_CROSS_ATTENTION_FREQ="${QFORMER_CROSS_ATTENTION_FREQ:-1}"

# Training hyperparameters (lower LR than stage 1)
SEED="${SEED:-42}"
EPOCHS="${EPOCHS:-1}"
PER_DEVICE_BATCH="${PER_DEVICE_BATCH:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
LR="${LR:-2e-5}"
RESAMPLER_LR="${RESAMPLER_LR:-1e-4}"
WD="${WD:-0.01}"
WARMUP="${WARMUP:-0.03}"
SCHED="${SCHED:-cosine}"
MAXLEN="${MAXLEN:-2048}"
LOG_STEPS="${LOG_STEPS:-10}"
SAVE_STEPS="${SAVE_STEPS:-1000}"
SAVE_TOTAL="${SAVE_TOTAL:-2}"
NUM_WORKERS="${NUM_WORKERS:-8}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

OUTDIR="${OUTPUT_ROOT}/llava-qformer-d${QFORMER_DEPTH}-l${QFORMER_LATENTS}-stage2"

# Sanity checks
if [[ ! -f "$DATA_PATH" ]]; then
  echo "Error: Instruction data not found at $DATA_PATH" >&2
  echo "Run: python LLaVA-NeXT/scripts/convert_caption_to_instruct.py --input /home/kezouke/Llava/coco/coco_train2014_llava.json --output $DATA_PATH" >&2
  exit 1
fi
if [[ ! -f "$STAGE1_ADAPTER" ]]; then
  echo "Error: Stage 1 adapter not found at $STAGE1_ADAPTER" >&2
  echo "Run stage 1 first." >&2
  exit 1
fi

mkdir -p "$OUTDIR"
echo "Stage 2: Instruction tuning -> $OUTDIR"

CMD=(python -m llava.train.train
  --model_name_or_path "$MODEL_PATH"
  --vision_tower "$VISION_TOWER"
  --data_path "$DATA_PATH"
  --image_folder "$IMAGE_FOLDER"
  --mm_resampler_type qformer
  --mm_qformer_depth "$QFORMER_DEPTH"
  --mm_qformer_latents "$QFORMER_LATENTS"
  --mm_qformer_cross_attention_freq "$QFORMER_CROSS_ATTENTION_FREQ"
  --mm_vision_resampler_lr "$RESAMPLER_LR"
  --mm_projector_type "linear"
  --mm_patch_merge_type flat
  --image_aspect_ratio square
  --mm_vision_select_layer -1
  --version llava_llama_3
  --freeze_backbone True
  --tune_mm_mlp_adapter True
  --tune_mm_vision_resampler True
  --pretrain_mm_mlp_adapter "$STAGE1_ADAPTER"
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
  --output_dir "$OUTDIR"
  --bf16 True
  --evaluation_strategy no
  --save_strategy steps
  --report_to none
  --dataloader_num_workers "$NUM_WORKERS"
  --lazy_preprocess True
  --overwrite_output_dir True
  --tf32 True
)

# User extras
if [[ -n "$EXTRA_ARGS" ]]; then
  # shellcheck disable=SC2206
  EXTRA_ARR=($EXTRA_ARGS)
  CMD+=("${EXTRA_ARR[@]}")
fi

printf 'Command: %q ' "${CMD[@]}" | tee "$OUTDIR/command.txt"
echo
"${CMD[@]}" 2>&1 | tee "$OUTDIR/train.log"

echo "Stage 2 finished. Output: $OUTDIR"
