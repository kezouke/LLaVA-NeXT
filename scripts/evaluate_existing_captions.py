#!/usr/bin/env python3
"""
Evaluate existing caption results without re-running inference.
This script loads pre-generated caption files and computes COCO metrics.
"""

import os
import json
import argparse
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap


def evaluate_existing_results(result_file, annotation_file):
    """
    Evaluate existing caption results using COCO metrics.
    
    Args:
        result_file: Path to JSON file with predictions (format: [{"image_id": int, "caption": str}, ...])
        annotation_file: Path to COCO annotation file
    
    Returns:
        Dictionary with evaluation metrics
    """
    print(f"Loading annotations from: {annotation_file}")
    coco = COCO(annotation_file)
    
    print(f"Loading results from: {result_file}")
    coco_result = coco.loadRes(result_file)
    
    print("Creating evaluator...")
    coco_eval = COCOEvalCap(coco, coco_result)
    coco_eval.params['image_id'] = coco_result.getImgIds()
    
    print("Running evaluation...")
    coco_eval.evaluate()
    
    print("\nEvaluation Scores:")
    for metric, score in coco_eval.eval.items():
        print(f"  {metric}: {score:.3f}")
    
    return coco_eval.eval


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate existing caption results without re-running inference"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="./results",
        help="Directory containing caption result files"
    )
    parser.add_argument(
        "--annotation_file",
        type=str,
        required=True,
        help="Path to COCO annotation file (e.g., captions_val2014.json)"
    )
    parser.add_argument(
        "--projectors",
        nargs="+",
        required=True,
        help="List of projector names to evaluate (e.g., mlp_1layer_8m mlp_1layer_12m)"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val2014",
        help="Dataset split name (default: val2014)"
    )
    
    args = parser.parse_args()
    
    # Check if annotation file exists
    if not os.path.exists(args.annotation_file):
        print(f"Error: Annotation file not found: {args.annotation_file}")
        return
    
    # Process each projector
    for proj in args.projectors:
        print(f"\n{'='*60}")
        print(f"Evaluating: {proj}")
        print(f"{'='*60}")
        
        # Construct result file path
        result_file = os.path.join(
            args.results_dir,
            f"captions_{args.split}_{proj}_results.json"
        )
        
        # Check if result file exists
        if not os.path.exists(result_file):
            print(f"Warning: Result file not found: {result_file}")
            print(f"Skipping {proj}")
            continue
        
        try:
            # Evaluate
            metrics = evaluate_existing_results(result_file, args.annotation_file)
            
            # Save metrics to file
            metrics_file = os.path.join(
                args.results_dir,
                f"coco_metrics_{args.split}_{proj}.json"
            )
            with open(metrics_file, 'w') as f:
                json.dump(metrics, f, indent=4)
            
            print(f"\nMetrics saved to: {metrics_file}")
            
        except Exception as e:
            print(f"Error evaluating {proj}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*60}")
    print("Evaluation complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()