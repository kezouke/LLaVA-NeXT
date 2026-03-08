#!/usr/bin/env bash
set -euo pipefail

# OPTIMIZED VERSION - Faster training with reduced epochs and data
# Use this for quick experiments or when time is limited

# Configuration
MODEL_PATH="${MODEL_PATH:-/data/ssd_storage/llama/}"
VISION_TOWER="${VISION_TOWER:-openai/clip-vit-large-patch14-336}"
DATA_PATH="${DATA_PATH:-/home/kezouke/Llava/coco/coco_train2014_llava.json}"
IMAGE_FOLDER="${IMAGE_FOLDER:-/home/kezouke/Llava/coco/train2014}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./checkpoints}"

# Q-Former fixed parameters
QFORMER_DEPTH=3
QFORMER_CROSS_ATTENTION_FREQ=1

# OPTIMIZED Training hyperparameters
SEED=42
EPOCHS="${EPOCHS:-1}"  # Reduced from 5 to 1 epoch for speed
PER_DEVICE_BATCH="${PER_DEVICE_BATCH:-8}"  # Increased from 4 to 8
GRAD_ACCUM="${GRAD_ACCUM:-4}"  # Reduced from 8 to 4 (effective batch stays same: 8*4=32)
LR=2e-4
RESAMPLER_LR=1e-3
WD=0.01
WARMUP=0.03

# Query count variants to sweep
QUERY_COUNTS=(8 16 32 64 128 256)

# Optional: Limit dataset size for faster iteration
MAX_SAMPLES="${MAX_SAMPLES:-}"  # Set to e.g. 50000 for faster training

# Function to train single variant
train_qformer_variant() {
    local num_queries=$1
    local outdir="${OUTPUT_ROOT}/llava-qformer-d${QFORMER_DEPTH}-l${num_queries}-train2014"
    
    echo "=========================================="
    echo "Training Q-Former with ${num_queries} queries"
    echo "Output: ${outdir}"
    echo "Epochs: ${EPOCHS}, Batch: ${PER_DEVICE_BATCH}, GradAccum: ${GRAD_ACCUM}"
    if [[ -n "$MAX_SAMPLES" ]]; then
        echo "Max samples: ${MAX_SAMPLES}"
    fi
    echo "=========================================="
    
    # Check if training is already complete
    if [[ -f "${outdir}/mm_projector.bin" ]]; then
        echo "Training already completed for ${num_queries} queries. Skipping."
        return
    fi

    # Build command
    CMD=(python -m llava.train.train
        --model_name_or_path "$MODEL_PATH"
        --vision_tower "$VISION_TOWER"
        --data_path "$DATA_PATH"
        --image_folder "$IMAGE_FOLDER"
        --mm_resampler_type qformer
        --mm_qformer_depth "$QFORMER_DEPTH"
        --mm_qformer_latents "$num_queries"
        --mm_qformer_cross_attention_freq "$QFORMER_CROSS_ATTENTION_FREQ"
        --mm_vision_resampler_lr "$RESAMPLER_LR"
        --mm_projector_type linear
        --mm_patch_merge_type flat
        --image_aspect_ratio square
        --mm_vision_select_layer -1
        --version plain
        --freeze_backbone True
        --tune_mm_mlp_adapter True
        --tune_mm_vision_resampler True
        --num_train_epochs "$EPOCHS"
        --per_device_train_batch_size "$PER_DEVICE_BATCH"
        --gradient_accumulation_steps "$GRAD_ACCUM"
        --learning_rate "$LR"
        --weight_decay "$WD"
        --warmup_ratio "$WARMUP"
        --lr_scheduler_type cosine
        --seed "$SEED"
        --attn_implementation sdpa
        --remove_unused_columns False
        --model_max_length 2048
        --logging_steps 10
        --save_steps 5000
        --save_total_limit 1
        --output_dir "$outdir"
        --bf16 True
        --evaluation_strategy no
        --save_strategy steps
        --report_to none
        --dataloader_num_workers 8
        --lazy_preprocess True
        --overwrite_output_dir True
        --tf32 True
        --dataloader_prefetch_factor 2
        --dataloader_persistent_workers True
    )
    
    # Add max_samples if specified
    if [[ -n "$MAX_SAMPLES" ]]; then
        CMD+=(--max_train_samples "$MAX_SAMPLES")
    fi
    
    "${CMD[@]}" 2>&1 | tee "${outdir}/train.log"
    
    echo "Completed training for ${num_queries} queries"
    echo ""
}

# Main execution
mkdir -p "$OUTPUT_ROOT"

echo "=========================================="
echo "OPTIMIZED Q-Former Query Sweep"
echo "Epochs: ${EPOCHS}"
echo "Batch size: ${PER_DEVICE_BATCH} (per device)"
echo "Gradient accumulation: ${GRAD_ACCUM}"
echo "Effective batch size: $((PER_DEVICE_BATCH * GRAD_ACCUM))"
if [[ -n "$MAX_SAMPLES" ]]; then
    echo "Max samples: ${MAX_SAMPLES}"
fi
echo "=========================================="

for num_queries in "${QUERY_COUNTS[@]}"; do
    # Create directory if it doesn't exist (for log file)
    mkdir -p "${OUTPUT_ROOT}/llava-qformer-d${QFORMER_DEPTH}-l${num_queries}-train2014"
    train_qformer_variant "$num_queries"
done

echo "=========================================="
echo "All Q-Former query sweep training completed!"
echo "=========================================="