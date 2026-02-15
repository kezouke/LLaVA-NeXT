#!/usr/bin/env bash
set -euo pipefail

# Test script for Q-Former with Auxiliary Loss and Partial Freezing

MODEL_PATH="/data/ssd_storage/llama/"
VISION_TOWER="openai/clip-vit-large-patch14-336"
DATA_PATH="/home/kezouke/Llava/coco/coco_train2014_llava.json"
IMAGE_FOLDER="/home/kezouke/Llava/coco/train2014"
OUTPUT_DIR="./checkpoints/test-qformer-aux-loss"

# Clean up previous test
rm -rf "$OUTPUT_DIR"

echo "Starting Q-Former Aux Loss Test..."

python -m llava.train.train \
    --model_name_or_path "$MODEL_PATH" \
    --vision_tower "$VISION_TOWER" \
    --data_path "$DATA_PATH" \
    --image_folder "$IMAGE_FOLDER" \
    --mm_projector_type "linear" \
    --mm_resampler_type "qformer" \
    --mm_qformer_depth 3 \
    --mm_qformer_latents 32 \
    --mm_qformer_use_aux_loss True \
    --mm_qformer_aux_loss_weight 0.1 \
    --mm_tune_qformer_layers "cross_attn_only" \
    --mm_patch_merge_type flat \
    --image_aspect_ratio square \
    --mm_vision_select_layer -1 \
    --freeze_backbone True \
    --tune_mm_mlp_adapter True \
    --tune_mm_vision_resampler True \
    --num_train_epochs 1 \
    --max_steps 100 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-4 \
    --weight_decay 0.0 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --save_steps 50 \
    --save_total_limit 1 \
    --output_dir "$OUTPUT_DIR" \
    --bf16 True \
    --report_to none \
    --dataloader_num_workers 4 \
    --lazy_preprocess True

echo "Test finished. Checking output directory..."
ls -l "$OUTPUT_DIR"