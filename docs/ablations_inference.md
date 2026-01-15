# Ablations Inference Script Documentation

This document describes how to use the `scripts/infer_ablations_batch.py` script to run inference on LLaVA-NeXT models with different projector configurations (ablations).

## Overview

The `infer_ablations_batch.py` script is designed to automate the process of evaluating multiple model checkpoints, specifically focusing on different projector architectures (e.g., `linear`, `mlp2x`, `mlp2x-res2x`, `pooler`). It runs inference on a sample of images from the COCO dataset and prints the model's predictions alongside the ground truth.

Key features:
- **Batch Processing:** Automatically iterates through a list of defined projector types.
- **Memory Management:** Spawns a separate subprocess for each model evaluation to ensure full GPU memory cleanup between runs.
- **Robustness:** Handles missing checkpoints or images gracefully.
- **Output Formatting:** Extracts the first sentence of the generated output for cleaner comparison.

## Prerequisites

Ensure you have the LLaVA-NeXT environment set up and the necessary dependencies installed.

## Configuration

Before running the script, you need to configure the paths and settings at the top of `scripts/infer_ablations_batch.py`:

```python
# --- Configuration ---
BASE_MODEL_PATH = "/path/to/your/base/llama/model"
DATA_PATH = "/path/to/coco/coco_train2014_llava.json"
IMAGE_FOLDER = "/path/to/coco/train2014"
CHECKPOINT_ROOT = "./checkpoints" 
DATASET_TAG = "train2014"
PROJECTORS = ["linear", "mlp2x", "mlp2x-res2x", "pooler"]
NUM_SAMPLES = 10
```

- `BASE_MODEL_PATH`: Path to the base LLM (e.g., Llama-2 or Llama-3).
- `DATA_PATH`: Path to the JSON file containing the dataset annotations (LLaVA format).
- `IMAGE_FOLDER`: Directory containing the images.
- `CHECKPOINT_ROOT`: Root directory where your trained checkpoints are stored.
- `DATASET_TAG`: Tag used in your checkpoint directory names (e.g., `llava-linear-train2014`).
- `PROJECTORS`: List of projector slugs to evaluate.
- `NUM_SAMPLES`: Number of random samples to evaluate per model.

## Directory Structure Assumption

The script assumes a specific directory structure for checkpoints:
`{CHECKPOINT_ROOT}/llava-{projector_slug}-{DATASET_TAG}/checkpoint-{step}`

It will automatically find the latest checkpoint within that directory.

## Usage

To run the script for all configured projectors:

```bash
python scripts/infer_ablations_batch.py
```

To run inference for a single specific projector (useful for debugging):

```bash
python scripts/infer_ablations_batch.py --projector linear
```

## How It Works

1.  **Parent Process:** The script starts in the parent process. It iterates through the `PROJECTORS` list.
2.  **Subprocess Spawning:** For each projector, it launches a new subprocess executing the same script but with the `--projector` argument.
3.  **Child Process:**
    -   Locates the latest checkpoint for the specified projector.
    -   Loads the model and tokenizer.
    -   **Crucial Step:** It manually reloads the projector weights (`mm_projector.bin`) to ensure they are correctly initialized, handling potential issues with `accelerate` or `deepspeed` saving mechanisms.
    -   Loads the dataset and selects random samples.
    -   Runs inference on the samples.
    -   Prints the image filename, ground truth (from the dataset), and the model's prediction.
4.  **Cleanup:** When the child process finishes, the OS reclaims all GPU memory, preventing OOM errors when loading the next model.

## Output

The script prints the progress and results to the console.

Example output:

```text
Launching subprocess for linear...

========================================
Evaluating Model with Projector: linear
========================================
Loading checkpoint: ./checkpoints/llava-linear-train2014/checkpoint-1000
...
[Sample 1/10] Image: COCO_train2014_000000057870.jpg
Ground Truth: A restaurant has modern wooden tables and chairs.
Prediction:   A restaurant with wooden tables and chairs.
...