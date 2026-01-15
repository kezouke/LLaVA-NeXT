import os
import json
import random
import argparse
import subprocess
import sys

# Add parent directory to path to allow importing llava modules
# This assumes the script is located in LLaVA-NeXT/scripts/
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from PIL import Image

# --- Configuration ---
# NOTE: Update these paths for your environment
BASE_MODEL_PATH = "/data/ssd_storage/llama/"
DATA_PATH = "/home/kezouke/Llava/coco/coco_train2014_llava.json"
IMAGE_FOLDER = "/home/kezouke/Llava/coco/train2014"
CHECKPOINT_ROOT = "./checkpoints" 
DATASET_TAG = "train2014"
PROJECTORS = ["linear", "mlp2x", "mlp2x-res2x", "pooler"]
NUM_SAMPLES = 10

def load_data(data_path):
    try:
        with open(data_path, 'r') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"Error: Data file not found at {data_path}")
        return []

def get_checkpoint_path(projector_slug, tag):
    model_dir_name = f"llava-{projector_slug}-{tag}"
    model_dir = os.path.join(CHECKPOINT_ROOT, model_dir_name)
    
    if not os.path.exists(model_dir):
        return None
        
    subdirs = [d for d in os.listdir(model_dir) if d.startswith('checkpoint-')]
    if not subdirs:
        return None
    
    try:
        subdirs.sort(key=lambda x: int(x.split('-')[1]))
    except ValueError:
        return None

    return os.path.join(model_dir, subdirs[-1])

def extract_first_sentence(text):
    # Check for the specific pattern where the model generates a fragment, then a newline, then the full sentence
    if "\n" in text:
        parts = text.split("\n")
        if len(parts) >= 2:
            first_part = parts[0].strip()
            second_part = parts[1].strip()
            
            # If the first part looks like a fragment (starts with lowercase) and the second part looks like a sentence
            if first_part and first_part[0].islower() and second_part and second_part[0].isupper():
                text = second_part
            # Also handle case where first part is empty or just punctuation
            elif not first_part and second_part:
                text = second_part

    # Find the earliest sentence terminator
    terminators = [".", "!", "?"]
    min_index = -1
    
    for terminator in terminators:
        try:
            index = text.index(terminator)
            if min_index == -1 or index < min_index:
                min_index = index
        except ValueError:
            continue
            
    if min_index != -1:
        return text[:min_index+1].strip()
    return text.strip()

def run_single_inference(projector):
    from llava.model.builder import load_pretrained_model
    from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path
    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
    from llava.conversation import conv_templates

    print(f"\n{'='*40}")
    print(f"Evaluating Model with Projector: {projector}")
    print(f"{'='*40}")
    
    ckpt_path = get_checkpoint_path(projector, DATASET_TAG)
    if not ckpt_path:
        print(f"Skipping {projector} due to missing checkpoint.")
        return
        
    print(f"Loading checkpoint: {ckpt_path}")
    
    try:
        model_name = get_model_name_from_path(ckpt_path)
        if "llama" not in model_name.lower():
            model_name = model_name + "-llama"
        
        # Use bfloat16 if available to match training and avoid overflow/underflow
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        print(f"Using dtype: {dtype}")

        tokenizer, model, image_processor, context_len = load_pretrained_model(
            model_path=ckpt_path,
            model_base=BASE_MODEL_PATH,
            model_name=model_name,
            device_map=device,
            torch_dtype=dtype,
            multimodal=True
        )
        
        # Explicitly cast the entire model to the correct dtype to ensure Vision Tower is compatible
        model.to(dtype=dtype)

        # Manually reload projector weights to fix initialization issues
        # This handles cases where builder.py fails to load them due to meta device or key mismatches
        print("Manually reloading projector weights...")
        mm_projector_path = os.path.join(ckpt_path, "mm_projector.bin")
        if os.path.exists(mm_projector_path):
            mm_projector_weights = torch.load(mm_projector_path, map_location="cpu")
            
            # Convert to correct dtype
            mm_projector_weights = {k: v.to(dtype) for k, v in mm_projector_weights.items()}
            
            # Load into model
            missing, unexpected = model.load_state_dict(mm_projector_weights, strict=False, assign=True)
            print(f"Reloaded projector. Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
            
            
            # Ensure projector is on the correct device
            model.get_model().mm_projector.to(device)
        else:
            print(f"Warning: mm_projector.bin not found at {mm_projector_path}")
        
        # Load Data
        data = load_data(DATA_PATH)
        data_with_images = [d for d in data if 'image' in d]
        
        # Use a fixed seed to ensure we get the same 'random' samples across subprocesses if needed,
        # OR just sample once in the parent and pass indices. 
        # For simplicity, let's just seed randomly here or use a fixed seed.
        random.seed(42)
        if len(data_with_images) >= NUM_SAMPLES:
            samples = random.sample(data_with_images, NUM_SAMPLES)
        else:
            samples = data_with_images

        for i, sample in enumerate(samples):
            image_file = sample['image']
            image_path = os.path.join(IMAGE_FOLDER, image_file)
            
            if not os.path.exists(image_path):
                print(f"\n[Sample {i+1}] Image file not found: {image_path}")
                continue
            
            try:
                image = Image.open(image_path).convert('RGB')
            except Exception as e:
                print(f"\n[Sample {i+1}] Error loading image: {e}")
                continue
            
            # Process Image
            image_tensor = process_images([image], image_processor, model.config)
            if type(image_tensor) is list:
                image_tensor = [img.to(device, dtype=dtype) for img in image_tensor]
            else:
                image_tensor = image_tensor.to(device, dtype=dtype)
            
            # Prepare Prompt
            # Use a simple prompt to avoid complex template issues
            prompt = DEFAULT_IMAGE_TOKEN
            
            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(device)
            
            try:
                with torch.inference_mode():
                    output_ids = model.generate(
                        input_ids,
                        images=image_tensor,
                        image_sizes=[image.size],
                        do_sample=False,
                        # temperature=0.2,
                        max_new_tokens=32,
                        use_cache=True,
                        pad_token_id=tokenizer.eos_token_id
                    )
                
                output = tokenizer.decode(output_ids[0, input_ids.shape[1]:], skip_special_tokens=True).strip()
                output = extract_first_sentence(output)
                
                ground_truth = "N/A"
                for msg in sample.get('conversations', []):
                    if msg['from'] == 'gpt':
                        ground_truth = msg['value']
                        break
                
                print(f"\n[Sample {i+1}/{len(samples)}] Image: {image_file}")
                print(f"Ground Truth: {ground_truth}")
                print(f"Prediction:   {output}")
            except Exception as e:
                print(f"\n[Sample {i+1}] Generation Error: {e}")

    except Exception as e:
        print(f"Error running model {projector}: {e}")
        import traceback
        traceback.print_exc()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--projector", type=str, help="Specific projector to run")
    args = parser.parse_args()

    if args.projector:
        # Child process mode
        run_single_inference(args.projector)
    else:
        # Parent process mode: spawn a subprocess for each projector
        # This ensures full GPU memory cleanup between runs
        for proj in PROJECTORS:
            print(f"Launching subprocess for {proj}...")
            cmd = [sys.executable, __file__, "--projector", proj]
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"Subprocess for {proj} failed with error: {e}")

if __name__ == "__main__":
    main()