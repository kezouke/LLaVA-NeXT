#!/usr/bin/env python3
"""
Convert COCO captions to LLaVA pretraining JSON format.

Usage:
  python LLaVA-NeXT/scripts/convert_coco_to_llava.py --coco /path/to/captions_train2014.json --output ./coco_projections_v1.json
"""

import argparse
import json
from typing import Optional

try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x


def convert_coco_to_llava(coco_annotation_path: str, output_path: str, id_prefix: str = "llava_pretrain_", max_samples: Optional[int] = None) -> None:
    """
    Convert COCO captions JSON to LLaVA format.

    Each COCO annotation becomes a separate LLaVA sample:
    {
      "id": "llava_pretrain_0",
      "image": "COCO_train2014_000000000009.jpg",
      "conversations": [
        {"from": "human", "value": "<image>\nA caption"},
        {"from": "gpt",   "value": "A caption"}
      ]
    }
    """
    with open(coco_annotation_path, "r") as f:
        coco = json.load(f)

    images = coco.get("images", [])
    annotations = coco.get("annotations", [])
    image_id_to_filename = {img["id"]: img["file_name"] for img in images}

    llava_data = []
    for ann in tqdm(annotations, desc="Converting annotations"):
        if max_samples is not None and len(llava_data) >= max_samples:
            break

        image_id = ann.get("image_id")
        caption = (ann.get("caption") or "").strip()
        image_filename = image_id_to_filename.get(image_id)
        if not image_filename:
            continue

        entry = {
            "id": f"{id_prefix}{len(llava_data)}",
            "image": image_filename,
            "conversations": [
                {"from": "human", "value": f"<image>\n{caption}"},
                {"from": "gpt", "value": caption},
            ],
        }
        llava_data.append(entry)

    with open(output_path, "w") as f:
        json.dump(llava_data, f, ensure_ascii=False, indent=2)

    print(f"Converted {len(llava_data)} samples to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert COCO captions to LLaVA pretraining JSON format.")
    parser.add_argument("--coco", required=True, help="Path to COCO captions JSON (e.g., captions_train2014.json)")
    parser.add_argument("--output", required=True, help="Output JSON path to write LLaVA-formatted data")
    parser.add_argument("--id-prefix", default="llava_pretrain_", help="Prefix for generated sample IDs")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional cap on the number of samples")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    convert_coco_to_llava(args.coco, args.output, id_prefix=args.id_prefix, max_samples=args.max_samples)


if __name__ == "__main__":
    main()