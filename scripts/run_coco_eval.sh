#!/usr/bin/env bash
set -euo pipefail

# COCO Evaluation Launcher for Projector Ablations

# --- Configuration ---
# Path to the base LLM (e.g., Llama-2-7b-hf)
MODEL_PATH="${MODEL_PATH:-/data/ssd_storage/llama/}"

# Root directory where checkpoints are saved (from training script)
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-./checkpoints}"

# COCO Paths
COCO_ROOT="${COCO_ROOT:-/home/kezouke/Llava/coco}"
VAL_IMAGE_FOLDER="${VAL_IMAGE_FOLDER:-$COCO_ROOT/val2014}"
TEST_IMAGE_FOLDER="${TEST_IMAGE_FOLDER:-$COCO_ROOT/test2014}"
VAL_ANNOTATION_FILE="${VAL_ANNOTATION_FILE:-$COCO_ROOT/annotations/captions_val2014.json}"

# Output directory for results
OUTPUT_DIR="${OUTPUT_DIR:-./results/coco_ablations}"

# Tag used during training (e.g., train2014) - used to find checkpoint folders
TRAIN_TAG="${TRAIN_TAG:-train2014}"

# Projectors to evaluate (space-separated)
PROJECTORS="${PROJECTORS:-linear mlp2x mlp2x-res2x pooler}"

# Max samples for debugging (leave empty for full dataset)
MAX_SAMPLES="${MAX_SAMPLES:-}"

# --- Execution ---

echo "Starting COCO Evaluation..."
echo "Model Path: $MODEL_PATH"
echo "Checkpoints: $CHECKPOINT_ROOT"
echo "Output Dir: $OUTPUT_DIR"
echo "Projectors: $PROJECTORS"

# 1. Run Validation Set (val2014) with Metrics
echo ""
echo "========================================================"
echo "Running Validation Set (val2014) with Metrics..."
echo "========================================================"

CMD_VAL=(python LLaVA-NeXT/scripts/eval_coco_projector_ablations.py
  --base_model_path "$MODEL_PATH"
  --checkpoint_root "$CHECKPOINT_ROOT"
  --image_folder "$VAL_IMAGE_FOLDER"
  --annotation_file "$VAL_ANNOTATION_FILE"
  --train_tag "$TRAIN_TAG"
  --split "val2014"
  --output_dir "$OUTPUT_DIR"
  --projectors $PROJECTORS
  --do_eval
)

if [[ -n "$MAX_SAMPLES" ]]; then
  CMD_VAL+=(--max_samples "$MAX_SAMPLES")
fi

echo "Running command: ${CMD_VAL[*]}"
"${CMD_VAL[@]}"


# 2. Run Test Set (test2014) - Predictions Only
echo ""
echo "========================================================"
echo "Running Test Set (test2014) - Predictions Only..."
echo "========================================================"

if [[ -d "$TEST_IMAGE_FOLDER" ]]; then
    CMD_TEST=(python LLaVA-NeXT/scripts/eval_coco_projector_ablations.py
      --base_model_path "$MODEL_PATH"
      --checkpoint_root "$CHECKPOINT_ROOT"
      --image_folder "$TEST_IMAGE_FOLDER"
      --train_tag "$TRAIN_TAG"
      --split "test2014"
      --output_dir "$OUTPUT_DIR"
      --projectors $PROJECTORS
    )

    if [[ -n "$MAX_SAMPLES" ]]; then
      CMD_TEST+=(--max_samples "$MAX_SAMPLES")
    fi

    echo "Running command: ${CMD_TEST[*]}"
    "${CMD_TEST[@]}"
else
    echo "Test image folder not found at $TEST_IMAGE_FOLDER. Skipping test set evaluation."
fi

echo ""
echo "All evaluations complete. Results saved to $OUTPUT_DIR"