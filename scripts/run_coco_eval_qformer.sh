#!/usr/bin/env bash
set -euo pipefail

# COCO Evaluation for Q-Former (stage 2 checkpoint)
# Generates predictions for val2014 (with metrics) and test2014 (predictions only)

MODEL_PATH="${MODEL_PATH:-/data/ssd_storage/llama/}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-./checkpoints}"
COCO_ROOT="${COCO_ROOT:-/home/kezouke/Llava/coco}"
VAL_IMAGE_FOLDER="${VAL_IMAGE_FOLDER:-$COCO_ROOT/val2014}"
TEST_IMAGE_FOLDER="${TEST_IMAGE_FOLDER:-$COCO_ROOT/test2014}"
VAL_ANNOTATION_FILE="${VAL_ANNOTATION_FILE:-$COCO_ROOT/annotations/captions_val2014.json}"
OUTPUT_DIR="${OUTPUT_DIR:-./results/coco_qformer}"

# Q-Former stage 2 projector slug (must match checkpoint directory name pattern)
PROJECTOR="${PROJECTOR:-qformer-d3-l32-stage2}"
TRAIN_TAG="${TRAIN_TAG:-}"

# Conversation template (llava_llama_3 for stage 2)
CONV_TEMPLATE="${CONV_TEMPLATE:-llava_llama_3}"

# Max samples for debugging (leave empty for full dataset)
MAX_SAMPLES="${MAX_SAMPLES:-}"

echo "Starting Q-Former COCO Evaluation..."
echo "Checkpoint: $CHECKPOINT_ROOT/llava-${PROJECTOR}${TRAIN_TAG:+-$TRAIN_TAG}"
echo "Output Dir: $OUTPUT_DIR"

# 1. Validation Set (val2014) with Metrics
echo ""
echo "========================================================"
echo "Running Validation Set (val2014) with Metrics..."
echo "========================================================"

CMD_VAL=(python LLaVA-NeXT/scripts/eval_coco_projector_ablations.py
  --base_model_path "$MODEL_PATH"
  --checkpoint_root "$CHECKPOINT_ROOT"
  --image_folder "$VAL_IMAGE_FOLDER"
  --annotation_file "$VAL_ANNOTATION_FILE"
  --train_tag "${TRAIN_TAG:-}"
  --split "val2014"
  --output_dir "$OUTPUT_DIR"
  --projectors "$PROJECTOR"
  --conv_template "$CONV_TEMPLATE"
  --do_eval
)

if [[ -n "$MAX_SAMPLES" ]]; then
  CMD_VAL+=(--max_samples "$MAX_SAMPLES")
fi

echo "Running: ${CMD_VAL[*]}"
"${CMD_VAL[@]}"

# 2. Test Set (test2014) - Predictions Only
echo ""
echo "========================================================"
echo "Running Test Set (test2014) - Predictions Only..."
echo "========================================================"

if [[ -d "$TEST_IMAGE_FOLDER" ]]; then
    CMD_TEST=(python LLaVA-NeXT/scripts/eval_coco_projector_ablations.py
      --base_model_path "$MODEL_PATH"
      --checkpoint_root "$CHECKPOINT_ROOT"
      --image_folder "$TEST_IMAGE_FOLDER"
      --train_tag "${TRAIN_TAG:-}"
      --split "test2014"
      --output_dir "$OUTPUT_DIR"
      --projectors "$PROJECTOR"
      --conv_template "$CONV_TEMPLATE"
    )

    if [[ -n "$MAX_SAMPLES" ]]; then
      CMD_TEST+=(--max_samples "$MAX_SAMPLES")
    fi

    echo "Running: ${CMD_TEST[*]}"
    "${CMD_TEST[@]}"
else
    echo "Test image folder not found at $TEST_IMAGE_FOLDER. Skipping."
fi

echo ""
echo "Q-Former evaluation complete. Results saved to $OUTPUT_DIR"
