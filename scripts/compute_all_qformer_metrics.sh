#!/usr/bin/env bash
set -euo pipefail

# Compute metrics for all Q-Former query sweep caption files

# Configuration
RESULTS_DIR="${RESULTS_DIR:-./results/qformer_query_sweep}"
ANNOTATION_FILE="${ANNOTATION_FILE:-/home/kezouke/Llava/coco/annotations/captions_val2014.json}"

# Q-Former variants
QFORMER_DEPTH=3
QUERY_COUNTS=(8 16 32 64 128)

echo "=========================================="
echo "Computing Metrics for Q-Former Query Sweep"
echo "Results directory: $RESULTS_DIR"
echo "=========================================="

# Check if results directory exists
if [ ! -d "$RESULTS_DIR" ]; then
    echo "Error: Results directory not found: $RESULTS_DIR"
    exit 1
fi

# Check if annotation file exists
if [ ! -f "$ANNOTATION_FILE" ]; then
    echo "Error: Annotation file not found: $ANNOTATION_FILE"
    exit 1
fi

# Process each variant
for num_queries in "${QUERY_COUNTS[@]}"; do
    projector_name="qformer-d${QFORMER_DEPTH}-l${num_queries}"
    caption_file="${RESULTS_DIR}/captions_val2014_${projector_name}_results.json"
    metrics_file="${RESULTS_DIR}/coco_metrics_val2014_${projector_name}.json"
    
    echo ""
    echo "Processing: ${projector_name}"
    echo "Caption file: ${caption_file}"
    
    if [ ! -f "$caption_file" ]; then
        echo "  ✗ Caption file not found, skipping"
        continue
    fi
    
    # Compute metrics
    python LLaVA-NeXT/scripts/compute_metrics_from_captions.py \
        --caption_file "$caption_file" \
        --annotation_file "$ANNOTATION_FILE" \
        --output_file "$metrics_file"
    
    echo "  ✓ Metrics saved to: ${metrics_file}"
done

echo ""
echo "=========================================="
echo "Metrics computation completed!"
echo "=========================================="
echo ""
echo "To analyze results, run:"
echo "  python LLaVA-NeXT/scripts/analyze_qformer_query_sweep_5variants.py"