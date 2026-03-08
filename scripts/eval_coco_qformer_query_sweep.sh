#!/usr/bin/env bash
set -euo pipefail

# Configuration
BASE_MODEL_PATH="${BASE_MODEL_PATH:-/data/ssd_storage/llama/}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-./checkpoints}"
IMAGE_FOLDER="${IMAGE_FOLDER:-/home/kezouke/Llava/coco/val2014}"
ANNOTATION_FILE="${ANNOTATION_FILE:-/home/kezouke/Llava/coco/annotations/captions_val2014.json}"
OUTPUT_DIR="${OUTPUT_DIR:-./results/qformer_query_sweep}"
TRAIN_TAG="${TRAIN_TAG:-train2014}"

# Q-Former variants
QFORMER_DEPTH=3
QUERY_COUNTS=(8 16 32 64 128)

# Create projector slugs
PROJECTORS=()
for num_queries in "${QUERY_COUNTS[@]}"; do
    PROJECTORS+=("qformer-d${QFORMER_DEPTH}-l${num_queries}")
done

echo "=========================================="
echo "Evaluating Q-Former Query Count Sweep"
echo "Variants: ${PROJECTORS[@]}"
echo "=========================================="

# Ensure output directory exists
mkdir -p "$OUTPUT_DIR"

# Run evaluation
python LLaVA-NeXT/scripts/eval_coco_projector_ablations.py \
    --base_model_path "$BASE_MODEL_PATH" \
    --checkpoint_root "$CHECKPOINT_ROOT" \
    --image_folder "$IMAGE_FOLDER" \
    --annotation_file "$ANNOTATION_FILE" \
    --train_tag "$TRAIN_TAG" \
    --split val2014 \
    --projectors "${PROJECTORS[@]}" \
    --output_dir "$OUTPUT_DIR" \
    --do_eval \
    --conv_template plain

echo "=========================================="
echo "Evaluation completed!"
echo "Results saved to: $OUTPUT_DIR"
echo "=========================================="