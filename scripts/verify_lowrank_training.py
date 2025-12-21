#!/usr/bin/env python3
"""
Verify LowRankProjector training setup:
- Ensures only projector parameters are trainable (if requested)
- Reports parameter counts and reduction vs a standard 2-layer MLP projector

Usage:
  python LLaVA-NeXT/scripts/verify_lowrank_training.py \
    --model_path /path/to/llava-or-llama-checkpoint \
    [--model_base /path/to/base-lm-if-lora-or-mm-adapter] \
    [--rank 64] \
    [--attn_implementation flash_attention_2] \
    [--enforce_projector_only True]

Notes:
- If you pass a language-model-only path, make sure it's a LLaVA-style checkpoint or
  pass a base model with --model_base as appropriate for LoRA/mm-projector-only checkpoints.
"""

import argparse
import os
import sys
import json
import torch

# Ensure 'llava' package path is importable when running from repo
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path


def fmt(n: int) -> str:
    return f"{n:,}"


def infer_projector_dims(model) -> tuple[int, int, int]:
    """
    Infer mm_hidden_size, hidden_size, rank from projector weights when not in config.
    Returns (mm_hidden_size, hidden_size, rank).
    """
    mm_hidden_size = getattr(model.config, "mm_hidden_size", None)
    hidden_size = getattr(model.config, "hidden_size", None)
    rank = getattr(model.config, "mm_low_rank_rank", None)

    down_w = None
    up_w = None
    for n, p in model.named_parameters():
        if "mm_projector.down_project.weight" in n:
            down_w = p
        elif "mm_projector.up_project.weight" in n:
            up_w = p

    if down_w is not None:
        # down_project: (rank, mm_hidden_size)
        if rank is None:
            rank = down_w.shape[0]
        if mm_hidden_size is None:
            mm_hidden_size = down_w.shape[1]

    if up_w is not None:
        # up_project: (hidden_size, rank)
        if hidden_size is None:
            hidden_size = up_w.shape[0]
        if rank is None:
            rank = up_w.shape[1]

    return int(mm_hidden_size), int(hidden_size), int(rank)


def enforce_projector_only_trainable(model) -> None:
    """
    Freeze all params except mm_projector.* to mimic training script behavior
    when tuning only the projector.
    """
    for n, p in model.named_parameters():
        if "mm_projector" in n:
            p.requires_grad_(True)
        else:
            p.requires_grad_(False)


def collect_param_stats(model) -> dict:
    total = 0
    trainable = 0
    projector = 0

    detailed = []

    for name, param in model.named_parameters():
        num = param.numel()
        total += num
        if "mm_projector" in name:
            projector += num
            detailed.append((name, num, param.requires_grad))
        if param.requires_grad:
            trainable += num

    return {
        "total": total,
        "trainable": trainable,
        "projector": projector,
        "details": detailed,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Verify LowRankProjector training configuration.")
    p.add_argument("--model_path", required=True, help="Path to model checkpoint (LLaVA or base LM).")
    p.add_argument("--model_base", default=None, help="Base model path if loading LoRA or projector-only checkpoints.")
    p.add_argument("--rank", type=int, default=64, help="Rank for LowRankProjector.")
    p.add_argument("--attn_implementation", default="flash_attention_2", help="Attention implementation for loading.")
    p.add_argument("--enforce_projector_only", type=lambda v: str(v).lower() in ['1','true','yes','y'], default=True,
                   help="If True, freeze all except projector to verify trainability.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    model_name = get_model_name_from_path(args.model_path)

    # Overwrite config to ensure we use the low-rank projector and known vision-token settings for pretrain
    overwrite_config = {
        "mm_projector_type": "low_rank",
        "mm_low_rank_rank": int(args.rank),
        "mm_use_im_patch_token": False,
        "mm_use_im_start_end": False,
    }

    # Load model
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=args.model_path,
        model_base=args.model_base,
        model_name=model_name,
        attn_implementation=args.attn_implementation,
        overwrite_config=overwrite_config,
    )

    # Optionally enforce only projector is trainable (freeze others)
    if args.enforce_projector_only:
        enforce_projector_only_trainable(model)

    # Collect stats
    stats = collect_param_stats(model)

    # Infer dims and compute reductions
    try:
        mm_hidden_size, hidden_size, rank = infer_projector_dims(model)
        standard_mlp_params = 2 * mm_hidden_size * hidden_size
        reduction = 100.0 * (1.0 - (stats["projector"] / standard_mlp_params))
    except Exception:
        mm_hidden_size = getattr(model.config, "mm_hidden_size", -1)
        hidden_size = getattr(model.config, "hidden_size", -1)
        rank = getattr(model.config, "mm_low_rank_rank", args.rank)
        standard_mlp_params = None
        reduction = None

    print("\nLowRankProjector Verification")
    print("-" * 60)
    print(f"Model path           : {args.model_path}")
    print(f"Model base           : {args.model_base}")
    print(f"Attention impl       : {args.attn_implementation}")
    print(f"Configured rank      : {rank}")
    print(f"mm_hidden_size       : {mm_hidden_size}")
    print(f"hidden_size          : {hidden_size}")
    print("-" * 60)
    print(f"Total params         : {fmt(stats['total'])}")
    print(f"Projector params     : {fmt(stats['projector'])}")
    print(f"Trainable params     : {fmt(stats['trainable'])}")
    if standard_mlp_params is not None and reduction is not None:
        print(f"Std MLP (2x) params  : {fmt(standard_mlp_params)}")
        print(f"Reduction vs Std MLP : {reduction:.2f}%")
    print("-" * 60)
    print("Projector parameter details (name | count | trainable):")
    for name, num, req in stats["details"]:
        print(f"  {name:60s} | {fmt(num):>12s} | {'TRAINABLE' if req else 'FROZEN'}")

    # Verification assertion: only projector should be trainable
    non_projector_trainable = stats["trainable"] - stats["projector"]
    if non_projector_trainable != 0:
        raise AssertionError(
            f"Found {fmt(non_projector_trainable)} non-projector trainable parameters. "
            f"Ensure freezing is correct (enforce_projector_only={args.enforce_projector_only})."
        )

    print("\n\u2713 Verification passed: Only LowRankProjector parameters are trainable.\n")


if __name__ == "__main__":
    main()