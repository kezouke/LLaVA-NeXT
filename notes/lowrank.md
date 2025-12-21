# Low-Rank Projector: Implementation, Training Procedure, and Reproducibility

This document provides a comprehensive description of the LowRankProjector we added to LLaVA-NeXT, including design rationale, implementation details, dataset preparation, training recipes, verification tools, and expected outputs. All steps are designed to be reproducible end-to-end.

References to code and scripts in this repository:
- [LLaVA-NeXT/llava/model/multimodal_projector/builder.py](../llava/model/multimodal_projector/builder.py)
- [LLaVA-NeXT/llava/train/train.py](../train/train.py)
- [LLaVA-NeXT/scripts/convert_coco_to_llava.py](../scripts/convert_coco_to_llava.py)
- [LLaVA-NeXT/scripts/v1_lowrank_pretrain.sh](../scripts/v1_lowrank_pretrain.sh)
- [LLaVA-NeXT/scripts/verify_lowrank_training.py](../scripts/verify_lowrank_training.py)
- [LLaVA-NeXT/scripts/zero2_torch_adamw.json](../scripts/zero2_torch_adamw.json)

---

## 1) Objective

Replace the standard dense MLP projector used to map vision features into the LLM’s embedding space with a low-rank factorized projector. The goal is to:
- Reduce trainable parameters in the projector by introducing a tunable rank r.
- Freeze the vision tower and the LLM backbone; train only the projector (parameter-efficient).
- Achieve stable pretraining on COCO captions while using the “plain” prompt format (no image special tokens).

---

## 2) Architecture

Low-rank projector defining the mapping from vision features (mm_hidden_size) to language embedding (hidden_size):

W ≈ A · B
- A: down-projects from mm_hidden_size → rank
- B: up-projects from rank → hidden_size

Forward path:
1) x ∈ R[B, T, mm_hidden_size]
2) down_project: R[mm_hidden_size → rank]
3) LayerNorm(rank)
4) GELU
5) up_project: R[rank → hidden_size]
6) y ∈ R[B, T, hidden_size]

Implementation additions:
- Class LowRankProjector with two linear layers (no bias), plus LayerNorm and GELU in-between.
- Registered projector_type="low_rank" in build_vision_projector.

Where:
- mm_hidden_size is the vision feature size (e.g., 1024 for CLIP-L/336).
- hidden_size is the language model embedding size (e.g., 4096 for Llama-3-8B style models).
- rank is configurable at runtime via --mm_low_rank_rank (default added to config: 64; we train with 256+).

---

## 3) Parameter Count and Trade-offs

Total parameters in LowRankProjector (ignoring LayerNorm):
- A: mm_hidden_size × rank
- B: rank × hidden_size
- Total ≈ rank × (mm_hidden_size + hidden_size)

Example (CLIP-L/336 → Llama-3-8B-like):
- mm_hidden_size = 1024
- hidden_size = 4096

r = 64:
- Params ≈ 64 × (1024 + 4096) = 64 × 5120 = 327,680 ≈ 0.33M

r = 256:
- Params ≈ 256 × 5120 = 1,310,720 ≈ 1.31M

r = 512:
- Params ≈ 512 × 5120 = 2,621,440 ≈ 2.62M

Rule of thumb:
- Larger r increases projector capacity and typically improves performance at the cost of more parameters and memory.
- r=256 is a balanced starting point for full COCO training on a single GPU with bf16.

---

## 4) Code Changes Summary

- Projector:
  - Implemented LowRankProjector and registered it in build_vision_projector.
  - File: [LLaVA-NeXT/llava/model/multimodal_projector/builder.py](../llava/model/multimodal_projector/builder.py)

- Config argument:
  - Added mm_low_rank_rank to ModelArguments for runtime control.
  - File: [LLaVA-NeXT/llava/train/train.py](../train/train.py)

- Utilities:
  - COCO converter to LLaVA JSON: [LLaVA-NeXT/scripts/convert_coco_to_llava.py](../scripts/convert_coco_to_llava.py)
  - Pretraining launcher (example): [LLaVA-NeXT/scripts/v1_lowrank_pretrain.sh](../scripts/v1_lowrank_pretrain.sh)
  - Verification tool: [LLaVA-NeXT/scripts/verify_lowrank_training.py](../scripts/verify_lowrank_training.py)
  - DeepSpeed config used in runs: [LLaVA-NeXT/scripts/zero2_torch_adamw.json](../scripts/zero2_torch_adamw.json)

---

## 5) Dataset Preparation (COCO Captions → LLaVA JSON)

We convert COCO captions (train2014/val2014) into LLaVA’s JSON format where each entry includes:
- "image": filename.jpg
- "conversations": [{"from": "human", "value": "<image>\n...caption..."}, {"from": "gpt", "value": "...caption..."}]

Conversion script:
- [LLaVA-NeXT/scripts/convert_coco_to_llava.py](../scripts/convert_coco_to_llava.py)

Example commands:
- Train split:
  python LLaVA-NeXT/scripts/convert_coco_to_llava.py \
    --coco /home/USER/Llava/coco/annotations/captions_train2014.json \
    --output /home/USER/Llava/coco/coco_train2014_llava.json \
    --id-prefix llava_train_

- Val split:
  python LLaVA-NeXT/scripts/convert_coco_to_llava.py \
    --coco /home/USER/Llava/coco/annotations/captions_val2014.json \
    --output /home/USER/Llava/coco/coco_val2014_llava.json \
    --id-prefix llava_val_

Training expects:
- --data_path pointing to the JSON
- --image_folder pointing to the images directory of that split

Example:
- data_path=/home/USER/Llava/coco/coco_train2014_llava.json
- image_folder=/home/USER/Llava/coco/train2014

---

## 6) Training Setup and Rationale

We train projector-only using bf16 and DeepSpeed ZeRO Stage 2 (Torch AdamW build).

Key choices:
- tune_mm_mlp_adapter=True → trains projector only
- freeze_backbone=True → freeze LLM
- vision_tower loaded but kept frozen
- version=plain → uses basic prompt format without image tokens
- mm_use_im_start_end=False and mm_use_im_patch_token=False via plain version defaults (compatible with projector pretraining)
- Gradient checkpointing enabled to reduce memory
- Cosine LR with warmup for stability
- Report_to none (disable W&B by default) for local smoke and full runs

DeepSpeed config used (Torch AdamW):
- [LLaVA-NeXT/scripts/zero2_torch_adamw.json](../scripts/zero2_torch_adamw.json)

Environment flags:
- DS_BUILD_OPS=0 → avoids compiling DeepSpeed’s customary CUDA ops if not needed or for simplified environments
- HF_ACCELERATE_USE_DEEPSPEED=1 → enable transformers’ accelerate integration with DeepSpeed

---

## 7) Smoke Test (Small Steps)

Observed logs representative of a short smoke run:
- Trainable parameters ~0.33 MB at rank=64 (matches math)
- “Only save projectors: True” in saving stage (projector-only checkpoint)
- Checkpoint created and training process completed successfully

Smoke-test style command:
DS_BUILD_OPS=0 HF_ACCELERATE_USE_DEEPSPEED=1 \
deepspeed LLaVA-NeXT/llava/train/train_mem.py \
  --deepspeed LLaVA-NeXT/scripts/zero2_torch_adamw.json \
  --model_name_or_path /data/ssd_storage/llama/ \
  --version plain \
  --data_path /path/to/sample_llava.json \
  --image_folder /path/to/images \
  --vision_tower openai/clip-vit-large-patch14-336 \
  --mm_projector_type low_rank --mm_low_rank_rank 64 \
  --tune_mm_mlp_adapter True --freeze_backbone True --bf16 True \
  --max_steps 5 \
  --per_device_train_batch_size 2 --gradient_accumulation_steps 1 \
  --evaluation_strategy no --save_strategy steps --save_steps 1000000 \
  --learning_rate 2e-3 --mm_projector_lr 2e-3 --weight_decay 0.0 \
  --warmup_ratio 0.03 --lr_scheduler_type cosine --logging_steps 1 --tf32 True \
  --model_max_length 2048 --gradient_checkpointing True \
  --dataloader_num_workers 4 --lazy_preprocess True --report_to none \
  --output_dir ./checkpoints/llava-lowrank-smoke

---

## 8) Full COCO Train with Rank 256 (Recommended Starting Point)

Ready-to-run command for single-GPU settings (adjust batch size to fit VRAM):

DS_BUILD_OPS=0 HF_ACCELERATE_USE_DEEPSPEED=1 \
deepspeed LLaVA-NeXT/llava/train/train_mem.py \
  --deepspeed LLaVA-NeXT/scripts/zero2_torch_adamw.json \
  --model_name_or_path /data/ssd_storage/llama/ \
  --version plain \
  --data_path /home/kezouke/Llava/coco/coco_train2014_llava.json \
  --image_folder /home/kezouke/Llava/coco/train2014 \
  --vision_tower openai/clip-vit-large-patch14-336 \
  --mm_projector_type low_rank --mm_low_rank_rank 256 \
  --tune_mm_mlp_adapter True --freeze_backbone True --bf16 True \
  --num_train_epochs 1 \
  --per_device_train_batch_size 2 --gradient_accumulation_steps 1 \
  --evaluation_strategy no --save_strategy steps --save_steps 2000 \
  --learning_rate 2e-3 --mm_projector_lr 2e-3 --weight_decay 0.0 \
  --warmup_ratio 0.03 --lr_scheduler_type cosine --logging_steps 10 --tf32 True \
  --model_max_length 2048 --gradient_checkpointing True \
  --dataloader_num_workers 16 --lazy_preprocess True --report_to none \
  --output_dir ./checkpoints/llava-lowrank-r256-train2014

Notes:
- Increase per_device_train_batch_size if you have more VRAM.
- For longer training, switch to --max_steps or more epochs and adjust save_steps.

---

## 9) Verification: Projector-Only Trainability and Parameter Reduction

Use the verification script to assert only projector parameters are trainable, and to print parameter counts and the reduction vs a standard 2-layer dense MLP:

Script:
- [LLaVA-NeXT/scripts/verify_lowrank_training.py](../scripts/verify_lowrank_training.py)

Example:
python LLaVA-NeXT/scripts/verify_lowrank_training.py \
  --model_path ./checkpoints/llava-lowrank-r256-train2014 \
  --model_base /data/ssd_storage/llama/ \
  --rank 256

Outputs:
- Total parameters, projector parameters, trainable parameters
- Reduction percentage vs standard dense projector (2 × mm_hidden_size × hidden_size)
- Raises AssertionError if any non-projector parameters are trainable

---

## 10) Practical Considerations and Known Messages

- torch.utils.checkpoint warnings:
  - “None of the inputs have requires_grad=True” can appear due to checkpointing wrappers; gradients still flow to the projector as long as projector params have requires_grad=True.
- bitsandbytes / BNB_CUDA_VERSION warnings:
  - Environment-specific; ensure compatible CUDA runtime if using bitsandbytes. Not required for projector-only bf16 training unless you explicitly deploy 4-/8-bit.
- DeepSpeed ops compilation:
  - DS_BUILD_OPS=0 avoids building custom CUDA ops; using the zero2_torch_adamw.json is a solid choice for projector-only runs.

---

## 11) Rank Selection Guidance

- r = 64:
  - ~0.33M params for projector
  - Good for smoke tests and very constrained hardware; may be under-capacity for high-quality alignment
- r = 128:
  - ~0.66M params; improved capacity with modest memory increase
- r = 256 (Recommended baseline):
  - ~1.31M params; balanced quality vs compute/memory for single-GPU bf16 training on COCO
- r = 512+:
  - ~2.62M params and higher; potentially better alignment but requires more VRAM and training time

Always check "Trainable parameters" in logs to ensure projector-only training. Expect an increase proportional to r.

---

## 12) Reproducibility Tips

- Fix random seeds if needed (torch, numpy, random) by extending training script flags or setting env vars; default recipes do not fix seeds.
- Persist exact pip/conda environment (requirements.txt or environment.yml).
- Record:
  - Model base path (model_name_or_path)
  - Data JSON path (data_path)
  - Image folder path (image_folder)
  - Vision tower identifier (openai/clip-vit-large-patch14-336)
  - Rank and all other training flags
  - DeepSpeed config file used

---

## 13) Future Work

- Compare different projector ranks (e.g., 128/256/512) on a held-out validation subset for caption alignment metrics or downstream tasks.
- Explore adding bias terms or residual adapters; experiment with different normalizations (RMSNorm) or nonlinearities.
- Evaluate with other vision towers (e.g., SigLIP) and alternative prompt formats.
- Consider curriculum (e.g., start with lower rank, then increase) or scheduling mm_projector_lr.

---

## 14) Quick Checklist

- Implemented and registered LowRankProjector → yes
- Exposed mm_low_rank_rank in model arguments → yes
- COCO → LLaVA JSON conversion done → yes
- Smoke test with small steps passed → yes
- Full COCO run (rank=256) command prepared → yes
- Verification script for projector-only trainability → yes

---

## 15) Appendix: Common Commands

Convert COCO:
- Train:
  python LLaVA-NeXT/scripts/convert_coco_to_llava.py \
    --coco /home/USER/Llava/coco/annotations/captions_train2014.json \
    --output /home/USER/Llava/coco/coco_train2014_llava.json \
    --id-prefix llava_train_

- Val:
  python LLaVA-NeXT/scripts/convert_coco_to_llava.py \
    --coco /home/USER/Llava/coco/annotations/captions_val2014.json \
    --output /home/USER/Llava/coco/coco_val2014_llava.json \
    --id-prefix llava_val_

Train projector (rank=256):
DS_BUILD_OPS=0 HF_ACCELERATE_USE_DEEPSPEED=1 \
deepspeed LLaVA-NeXT/llava/train/train_mem.py \
  --deepspeed LLaVA-NeXT/scripts/zero2_torch_adamw.json \
  --model_name_or_path /data/ssd_storage/llama/ \
  --version plain \
  --data_path /home/USER/Llava/coco/coco_train2014_llava.json \
  --image_folder /home/USER/Llava/coco/train2014 \
  --vision_tower openai/clip-vit-large-patch14-336 \
  --mm_projector_type low_rank --mm_low_rank_rank 256 \
  --tune_mm_mlp_adapter True --freeze_backbone True --bf16 True \
  --num_train_epochs 1 \
  --per_device_train_batch_size 2 --gradient_accumulation_steps 1 \
  --evaluation_strategy no --save_strategy steps --save_steps 2000 \
  --learning_rate 2e-3 --mm_projector_lr 2e-3 --weight_decay 0.0 \
  --warmup_ratio 0.03 --lr_scheduler_type cosine --logging_steps 10 --tf32 True \
  --model_max_length 2048 --gradient_checkpointing True \
  --dataloader_num_workers 16 --lazy_preprocess True --report_to none \
  --output_dir ./checkpoints/llava-lowrank-r256-train2014

Verify projector-only:
python LLaVA-NeXT/scripts/verify_lowrank_training.py \
  --model_path ./checkpoints/llava-lowrank-r256-train2014 \
  --model_base /data/ssd_storage/llama/ \
  --rank 256

---

End of document.
