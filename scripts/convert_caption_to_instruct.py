#!/usr/bin/env python3
"""Convert LLaVA-Pretrain caption data to instruction-tuning format.

Input format (caption in both turns):
  {"from": "human", "value": "<image>\nA cat on a couch"},
  {"from": "gpt",   "value": "A cat on a couch"}

Output format (instruction in human turn, caption only in gpt turn):
  {"from": "human", "value": "<image>\nDescribe this image."},
  {"from": "gpt",   "value": "A cat on a couch"}

Usage:
  python convert_caption_to_instruct.py \
    --input /path/to/coco_train2014_llava.json \
    --output /path/to/coco_train2014_instruct.json
"""

import argparse
import json
import random

INSTRUCT_PROMPTS = [
    "Describe this image.",
    "What do you see in this image?",
    "Provide a brief description of the given image.",
    "What is shown in this image?",
    "Describe the image briefly.",
    "Give a short description of this image.",
    "What does this image depict?",
    "Summarize the content of this image.",
    "What is happening in this image?",
    "Briefly describe what you see.",
]


def convert(input_path: str, output_path: str, seed: int = 42):
    random.seed(seed)

    with open(input_path, "r") as f:
        data = json.load(f)

    converted = []
    for sample in data:
        convs = sample.get("conversations", [])
        if len(convs) < 2:
            continue

        # Extract the caption from the gpt turn
        caption = convs[1]["value"]

        # Replace human turn with <image> + random instruction
        prompt = random.choice(INSTRUCT_PROMPTS)
        new_convs = [
            {"from": "human", "value": f"<image>\n{prompt}"},
            {"from": "gpt", "value": caption},
        ]

        new_sample = {k: v for k, v in sample.items() if k != "conversations"}
        new_sample["conversations"] = new_convs
        converted.append(new_sample)

    with open(output_path, "w") as f:
        json.dump(converted, f, indent=2)

    print(f"Converted {len(converted)} samples: {input_path} -> {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert caption data to instruction format")
    parser.add_argument("--input", required=True, help="Path to input caption JSON")
    parser.add_argument("--output", required=True, help="Path to output instruction JSON")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for prompt selection")
    args = parser.parse_args()
    convert(args.input, args.output, args.seed)
