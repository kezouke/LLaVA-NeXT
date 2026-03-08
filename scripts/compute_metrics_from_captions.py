#!/usr/bin/env python3
"""
Compute COCO metrics from existing caption JSON files
Use this when you have caption predictions but need to compute metrics
"""

import json
import os
import argparse
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap

def compute_metrics(caption_file, annotation_file, output_file=None):
    """
    Compute COCO captioning metrics from a caption file
    
    Args:
        caption_file: Path to JSON file with predictions (format: [{"image_id": int, "caption": str}, ...])
        annotation_file: Path to COCO annotations file
        output_file: Optional path to save metrics JSON
    """
    print(f"Loading annotations from: {annotation_file}")
    coco = COCO(annotation_file)
    
    print(f"Loading predictions from: {caption_file}")
    coco_result = coco.loadRes(caption_file)
    
    print("Computing metrics...")
    coco_eval = COCOEvalCap(coco, coco_result)
    
    # Evaluate on the images in the results file
    coco_eval.params['image_id'] = coco_result.getImgIds()
    
    # Run evaluation
    coco_eval.evaluate()
    
    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    for metric, score in coco_eval.eval.items():
        print(f"{metric:12s}: {score:.3f}")
    print("="*50)
    
    # Save to file if requested
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(coco_eval.eval, f, indent=4)
        print(f"\nMetrics saved to: {output_file}")
    
    return coco_eval.eval

def main():
    parser = argparse.ArgumentParser(description="Compute COCO metrics from caption files")
    parser.add_argument("--caption_file", type=str, required=True,
                       help="Path to caption predictions JSON file")
    parser.add_argument("--annotation_file", type=str, required=True,
                       help="Path to COCO annotations file")
    parser.add_argument("--output_file", type=str, default=None,
                       help="Path to save metrics JSON (optional)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.caption_file):
        print(f"Error: Caption file not found: {args.caption_file}")
        return
    
    if not os.path.exists(args.annotation_file):
        print(f"Error: Annotation file not found: {args.annotation_file}")
        return
    
    compute_metrics(args.caption_file, args.annotation_file, args.output_file)

if __name__ == "__main__":
    main()