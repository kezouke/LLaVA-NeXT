#!/usr/bin/env bash
set -euo pipefail

# FAST EVALUATION - Uses subset of validation set for quick results
# For full evaluation, use eval_coco_qformer_query_sweep_5variants.sh

# Configuration
BASE_MODEL_PATH="${BASE_MODEL_PATH:-/data/ssd_storage/llama/}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-./checkpoints}"
IMAGE_FOLDER="${IMAGE_FOLDER:-/home/kezouke/Llava/coco/val2014}"
ANNOTATION_FILE="${ANNOTATION_FILE:-/home/kezouke/Llava/coco/annotations/captions_val2014.json}"
OUTPUT_DIR="${OUTPUT_DIR:-./results/qformer_query_sweep}"
TRAIN_TAG="${TRAIN_TAG:-train2014}"

# Q-Former variants (excluding 256)
QFORMER_DEPTH=3
QUERY_COUNTS=(8 16 32 64 128)

# SPEEDUP: Limit number of samples for faster evaluation
# Full val set has 40,504 images
# Using 5,000 images gives statistically significant results much faster
MAX_SAMPLES="${MAX_SAMPLES:-5000}"

# Create projector slugs
PROJECTORS=()
for num_queries in "${QUERY_COUNTS[@]}"; do
    PROJECTORS+=("qformer-d${QFORMER_DEPTH}-l${num_queries}")
done

echo "=========================================="
echo "FAST Q-Former Query Count Sweep Evaluation"
echo "Variants: ${PROJECTORS[@]}"
echo "Max samples: ${MAX_SAMPLES} (out of 40,504)"
echo "=========================================="

# Ensure output directory exists
mkdir -p "$OUTPUT_DIR"

# Run evaluation with max_samples limit
python LLaVA-NeXT/scripts/eval_coco_projector_ablations.py \
    --base_model_path "$BASE_MODEL_PATH" \
    --checkpoint_root "$CHECKPOINT_ROOT" \
    --image_folder "$IMAGE_FOLDER" \
    --annotation_file "$ANNOTATION_FILE" \
    --train_tag "$TRAIN_TAG" \
    --split val2014 \
    --projectors "${PROJECTORS[@]}" \
    --output_dir "$OUTPUT_DIR" \
    --max_samples "$MAX_SAMPLES" \
    --do_eval \
    --conv_template plain

echo "=========================================="
echo "Fast evaluation completed!"
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Note: This used ${MAX_SAMPLES} samples for speed."
echo "For full evaluation (40,504 samples), run:"
echo "  bash LLaVA-NeXT/scripts/eval_coco_qformer_query_sweep_5variants.sh"
echo "=========================================="