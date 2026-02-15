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

# Updated to test Q-Former with auxiliary loss
# Format: "qformer-aux-loss-d{depth}-l{latents}"
PROJECTORS = ["qformer-d3-l32"]
NUM_SAMPLES = 10  # Number of samples to test

def load_data(data_path):
    try:
        with open(data_path, 'r') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"Error: Data file not found at {data_path}")
        return []

def get_checkpoint_path(projector_slug, tag):
    """
    Get the latest checkpoint path for a given projector configuration.
    For Q-Former with aux loss, the directory format is:
    llava-qformer-aux-loss-d{depth}-l{latents}-{tag}
    """
    model_dir_name = f"llava-{projector_slug}-{tag}"
    model_dir = os.path.join(CHECKPOINT_ROOT, model_dir_name)
    
    print(f"Looking for checkpoint in: {model_dir}")
    
    if not os.path.exists(model_dir):
        print(f"Directory not found: {model_dir}")
        return None
        
    subdirs = [d for d in os.listdir(model_dir) if d.startswith('checkpoint-')]
    if not subdirs:
        print(f"No checkpoint subdirectories found in {model_dir}")
        return None
    
    try:
        subdirs.sort(key=lambda x: int(x.split('-')[1]))
    except ValueError:
        print(f"Could not parse checkpoint numbers from: {subdirs}")
        return None

    checkpoint_path = os.path.join(model_dir, subdirs[-1])
    print(f"Using latest checkpoint: {checkpoint_path}")
    return checkpoint_path

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

def run_single_inference(projector, checkpoint_path=None):
    from llava.model.builder import load_pretrained_model
    from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path
    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
    from llava.conversation import conv_templates

    print(f"\n{'='*40}")
    print(f"Evaluating Model with Projector: {projector}")
    print(f"{'='*40}")
    
    if checkpoint_path:
        ckpt_path = checkpoint_path
    else:
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

        # Check if this is a full checkpoint (with safetensors) or just projector weights
        has_safetensors = os.path.exists(os.path.join(ckpt_path, "model.safetensors.index.json"))
        has_mm_projector_bin = os.path.exists(os.path.join(ckpt_path, "mm_projector.bin"))
        
        print(f"Checkpoint format: safetensors={has_safetensors}, mm_projector.bin={has_mm_projector_bin}")
        
        if has_safetensors and not has_mm_projector_bin:
            # Full checkpoint with safetensors - load directly without model_base
            print("Loading full model from safetensors checkpoint...")
            tokenizer, model, image_processor, context_len = load_pretrained_model(
                model_path=ckpt_path,
                model_base=None,  # Don't use base model, load everything from checkpoint
                model_name=model_name,
                device_map=device,
                torch_dtype=dtype,
                multimodal=True
            )
        else:
            # Old format with mm_projector.bin or need to load from base
            print("Loading model with base model + projector weights...")
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
        model.eval()

        print(f"Model Config mm_resampler_type: {getattr(model.config, 'mm_resampler_type', 'None')}")
        print(f"Model Config mm_projector_type: {getattr(model.config, 'mm_projector_type', 'None')}")
        print(f"Model Config mm_qformer_use_aux_loss: {getattr(model.config, 'mm_qformer_use_aux_loss', 'None')}")
        print(f"Model Config mm_qformer_aux_loss_weight: {getattr(model.config, 'mm_qformer_aux_loss_weight', 'None')}")

        # Check if weights were loaded properly from safetensors
        # The builder.py should have loaded them, but let's verify
        print("\nVerifying loaded weights...")
        
        # Check if mm_projector.bin exists (old format)
        mm_projector_path = os.path.join(ckpt_path, "mm_projector.bin")
        
        # Fallback 1: mm_projector/checkpoint-XX.bin (saved by train.py when saving by step)
        if not os.path.exists(mm_projector_path):
            ckpt_folder = os.path.basename(os.path.normpath(ckpt_path))
            if ckpt_folder.startswith("checkpoint-"):
                parent_dir = os.path.dirname(ckpt_path)
                alt_path = os.path.join(parent_dir, "mm_projector", f"{ckpt_folder}.bin")
                if os.path.exists(alt_path):
                    print(f"Found mm_projector at: {alt_path}")
                    mm_projector_path = alt_path
        # Fallback 2: parent directory mm_projector.bin (final save at run root)
        if not os.path.exists(mm_projector_path):
            parent_mm_projector_path = os.path.join(os.path.dirname(ckpt_path), "mm_projector.bin")
            if os.path.exists(parent_mm_projector_path):
                print(f"Found mm_projector.bin in parent directory: {parent_mm_projector_path}")
                mm_projector_path = parent_mm_projector_path

        if os.path.exists(mm_projector_path):
            print(f"Loading weights from {mm_projector_path}...")
            mm_projector_weights = torch.load(mm_projector_path, map_location="cpu")
            
            # Convert to correct dtype
            mm_projector_weights = {k: v.to(dtype) for k, v in mm_projector_weights.items()}

            # Remove 'module.' prefix if present
            if any(k.startswith("module.") for k in mm_projector_weights):
                mm_projector_weights = {k[7:] if k.startswith("module.") else k: v for k, v in mm_projector_weights.items()}

            # Handle potential key mismatches for vision_resampler
            new_mm_projector_weights = {}
            for k, v in mm_projector_weights.items():
                new_mm_projector_weights[k] = v
                if k.startswith("model.vision_resampler."):
                    new_mm_projector_weights[k[6:]] = v
                elif k.startswith("vision_resampler."):
                    new_mm_projector_weights[f"model.{k}"] = v
            mm_projector_weights = new_mm_projector_weights
            
            # Debug: Print keys in mm_projector_weights
            print(f"Keys in mm_projector.bin: {list(mm_projector_weights.keys())[:5]} ... (Total: {len(mm_projector_weights)})")

            # Debug: Check value in DICT
            ln_key = "model.vision_resampler.ln_vision.weight"
            if ln_key in mm_projector_weights:
                w = mm_projector_weights[ln_key]
                print(f"DEBUG: Dict ln_vision weight: mean={w.mean().item():.6f}, std={w.std().item():.6f}")
            else:
                print(f"DEBUG: {ln_key} NOT in dict")
            
            # Check Q-Former query weight in checkpoint
            qformer_key = "model.vision_resampler.Qformer.bert.encoder.layer.0.crossattention.self.query.weight"
            checkpoint_qf_weight = None
            if qformer_key in mm_projector_weights:
                checkpoint_qf_weight = mm_projector_weights[qformer_key]
                print(f"DEBUG: Dict Qformer query weight: mean={checkpoint_qf_weight.mean().item():.6f}, std={checkpoint_qf_weight.std().item():.6f}")
                print(f"DEBUG: Dict Qformer query first 3: {checkpoint_qf_weight.flatten()[:3].tolist()}")
            else:
                # Try without model. prefix
                qformer_key2 = "vision_resampler.Qformer.bert.encoder.layer.0.crossattention.self.query.weight"
                if qformer_key2 in mm_projector_weights:
                    checkpoint_qf_weight = mm_projector_weights[qformer_key2]
                    print(f"DEBUG: Dict Qformer query weight: mean={checkpoint_qf_weight.mean().item():.6f}, std={checkpoint_qf_weight.std().item():.6f}")
                    print(f"DEBUG: Dict Qformer query first 3: {checkpoint_qf_weight.flatten()[:3].tolist()}")
                else:
                    print(f"DEBUG: Q-Former query weight NOT in checkpoint! Keys containing 'crossattention.self.query': {[k for k in mm_projector_weights.keys() if 'crossattention.self.query' in k][:3]}")

            # Debug: Check specific weights BEFORE reload
            if hasattr(model.get_model(), "vision_resampler"):
                resampler = model.get_model().vision_resampler
                if hasattr(resampler, "ln_vision"):
                    print(f"DEBUG: ln_vision weight BEFORE: mean={resampler.ln_vision.weight.mean().item():.6f}, std={resampler.ln_vision.weight.std().item():.6f}")
                if hasattr(resampler, "Qformer"):
                    q_weight = resampler.Qformer.bert.encoder.layer[0].crossattention.self.query.weight
                    print(f"DEBUG: Qformer query weight BEFORE: mean={q_weight.mean().item():.6f}, std={q_weight.std().item():.6f}")
                    # High precision check - first 3 elements
                    print(f"DEBUG: Qformer query BEFORE first 3: {q_weight.flatten()[:3].tolist()}")
                    before_data_ptr = q_weight.data_ptr()
                    print(f"DEBUG: Qformer query BEFORE data_ptr: {before_data_ptr}")

            # Load DIRECTLY into submodules to avoid key matching issues with full model state_dict
            # This is more reliable than model.load_state_dict() for nested modules
            
            # Extract mm_projector weights (remove prefix)
            projector_weights = {}
            for k, v in mm_projector_weights.items():
                if "mm_projector" in k:
                    # Get the key relative to mm_projector module
                    if k.startswith("model.mm_projector."):
                        new_key = k.replace("model.mm_projector.", "")
                        projector_weights[new_key] = v
                    elif k.startswith("mm_projector."):
                        new_key = k.replace("mm_projector.", "")
                        projector_weights[new_key] = v
            
            # Extract vision_resampler weights (remove prefix)
            resampler_weights = {}
            for k, v in mm_projector_weights.items():
                if "vision_resampler" in k:
                    # Get the key relative to vision_resampler module
                    if k.startswith("model.vision_resampler."):
                        new_key = k.replace("model.vision_resampler.", "")
                        resampler_weights[new_key] = v
                    elif k.startswith("vision_resampler."):
                        new_key = k.replace("vision_resampler.", "")
                        resampler_weights[new_key] = v
            
            print(f"Extracted {len(projector_weights)} projector weights, {len(resampler_weights)} resampler weights")
            
            # Debug: Check if Q-Former query weight is in extracted resampler_weights
            qf_query_key = "Qformer.bert.encoder.layer.0.crossattention.self.query.weight"
            if qf_query_key in resampler_weights:
                w = resampler_weights[qf_query_key]
                print(f"DEBUG: Extracted Qformer query: mean={w.mean().item():.6f}, std={w.std().item():.6f}")
            else:
                print(f"DEBUG: '{qf_query_key}' NOT in resampler_weights!")
                print(f"DEBUG: Sample resampler_weights keys: {list(resampler_weights.keys())[:5]}")
            
            # Load directly into mm_projector
            if projector_weights and hasattr(model.get_model(), 'mm_projector'):
                proj_result = model.get_model().mm_projector.load_state_dict(projector_weights, strict=False, assign=True)
                print(f"mm_projector: missing={len(proj_result.missing_keys)}, unexpected={len(proj_result.unexpected_keys)}")
            
            # Load directly into vision_resampler
            if resampler_weights and hasattr(model.get_model(), 'vision_resampler'):
                resampler_result = model.get_model().vision_resampler.load_state_dict(resampler_weights, strict=False, assign=True)
                print(f"vision_resampler: missing={len(resampler_result.missing_keys)}, unexpected={len(resampler_result.unexpected_keys)}")
                if resampler_result.missing_keys:
                    print(f"  Missing: {resampler_result.missing_keys[:5]}...")
                if resampler_result.unexpected_keys:
                    print(f"  Unexpected: {resampler_result.unexpected_keys[:5]}...")

            # Debug: Check specific weights AFTER reload
            if hasattr(model.get_model(), "vision_resampler"):
                resampler = model.get_model().vision_resampler
                if hasattr(resampler, "ln_vision"):
                    print(f"DEBUG: ln_vision weight AFTER: mean={resampler.ln_vision.weight.mean().item():.6f}, std={resampler.ln_vision.weight.std().item():.6f}")
                if hasattr(resampler, "Qformer"):
                    q_weight = resampler.Qformer.bert.encoder.layer[0].crossattention.self.query.weight
                    print(f"DEBUG: Qformer query weight AFTER: mean={q_weight.mean().item():.6f}, std={q_weight.std().item():.6f}")
                    # High precision check - first 3 elements
                    print(f"DEBUG: Qformer query AFTER first 3: {q_weight.flatten()[:3].tolist()}")
                    after_data_ptr = q_weight.data_ptr()
                    print(f"DEBUG: Qformer query AFTER data_ptr: {after_data_ptr}")
                    # Check if tensor was actually replaced (assign=True should change data_ptr)
                    if 'before_data_ptr' in dir():
                        if after_data_ptr != before_data_ptr:
                            print("DEBUG: SUCCESS - tensor was replaced (data_ptr changed)")
                        else:
                            print("DEBUG: WARNING - tensor NOT replaced (same data_ptr) - assign=True may not be working!")

            # Verify weights are not all zeros
            if hasattr(model.get_model(), "mm_projector"):
                proj_weight = next(model.get_model().mm_projector.parameters())
                print(f"Projector weight mean: {proj_weight.mean().item()}, std: {proj_weight.std().item()}")
                if proj_weight.abs().sum() == 0:
                    print("WARNING: Projector weights are all zero!")

            # Ensure projector is on the correct device
            model.get_model().mm_projector.to(device)
            
            # Ensure vision resampler (e.g., Q-Former) is on the correct device
            if hasattr(model.get_model(), 'vision_resampler'):
                model.get_model().vision_resampler.to(device)

            # Ensure vision tower is on the correct device
            if hasattr(model.get_model(), 'get_vision_tower'):
                vision_tower = model.get_model().get_vision_tower()
                if vision_tower is not None:
                    vision_tower.to(device=device, dtype=dtype)
        else:
            # Weights should already be loaded from safetensors by builder.py
            print("No mm_projector.bin found - weights should be loaded from safetensors")
            
            # Verify that weights are actually loaded and not random
            if hasattr(model.get_model(), 'vision_resampler'):
                resampler = model.get_model().vision_resampler
                if hasattr(resampler, 'ln_vision'):
                    ln_weight = resampler.ln_vision.weight
                    print(f"ln_vision weight: mean={ln_weight.mean().item():.6f}, std={ln_weight.std().item():.6f}")
                    if ln_weight.abs().sum() == 0:
                        print("WARNING: ln_vision weights are all zero!")
                
                if hasattr(resampler, 'query_tokens'):
                    qt = resampler.query_tokens
                    print(f"Q-Former query_tokens: mean={qt.mean().item():.6f}, std={qt.std().item():.6f}")
                    if torch.isnan(qt).any():
                        print("CRITICAL: Q-Former query_tokens contain NaNs!")

                if hasattr(resampler, 'Qformer'):
                    q_weight = resampler.Qformer.bert.encoder.layer[0].crossattention.self.query.weight
                    print(f"Q-Former query weight: mean={q_weight.mean().item():.6f}, std={q_weight.std().item():.6f}")
                    if q_weight.abs().sum() == 0:
                        print("WARNING: Q-Former weights are all zero!")
                    if torch.isnan(q_weight).any():
                        print("CRITICAL: Q-Former weights contain NaNs!")
                
                if hasattr(resampler, 'aux_head'):
                    aux_weight = resampler.aux_head.weight
                    print(f"Aux head weight: mean={aux_weight.mean().item():.6f}, std={aux_weight.std().item():.6f}")
                    if aux_weight.abs().sum() == 0:
                        print("WARNING: Aux head weights are all zero!")
            
            if hasattr(model.get_model(), 'mm_projector'):
                proj_weight = next(model.get_model().mm_projector.parameters())
                print(f"Projector weight: mean={proj_weight.mean().item():.6f}, std={proj_weight.std().item():.6f}")
                if proj_weight.abs().sum() == 0:
                    print("WARNING: Projector weights are all zero!")
        
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
            # Use the correct conversation template for Llama-3
            conv = conv_templates["llava_llama_3"].copy()
            # IMPORTANT: Inject the tokenizer into the conversation object so apply_chat_template works
            conv.tokenizer = tokenizer
            conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\nDescribe this image.")
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            print(f"DEBUG: Prompt: {prompt}")
            
            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(device)
            
            # Debug: Check image features for EVERY sample to see if they change
            try:
                with torch.inference_mode():
                    # 0. Check Input IDs (mask image token for decode — vocab indices must be non-negative)
                    if i == 0:
                        print(f"DEBUG: Input IDs: {input_ids}")
                        safe_ids = input_ids[0].clone()
                        safe_ids[safe_ids == IMAGE_TOKEN_INDEX] = tokenizer.unk_token_id if tokenizer.unk_token_id is not None else 0
                        print(f"DEBUG: Decoded Input: {tokenizer.decode(safe_ids)}")
                        print(f"DEBUG: IMAGE_TOKEN_INDEX: {IMAGE_TOKEN_INDEX}")

                    # 1. Check Vision Tower Output
                    vision_tower = model.get_model().get_vision_tower()
                    image_features = vision_tower(image_tensor)
                    feat_sum = image_features.sum().item()
                    feat_mean = image_features.mean().item()
                    feat_std = image_features.std().item()
                    print(f"DEBUG [Sample {i+1}]: Vision Tower Sum: {feat_sum:.4f}, Mean: {feat_mean:.6f}, Std: {feat_std:.6f}")
                    
                    # 2. Check Resampler Output (Q-Former with aux loss)
                    if hasattr(model.get_model(), "vision_resampler"):
                        # Q-Former returns tuple (features, aux_loss) when use_aux_loss=True
                        resampler_output = model.get_model().vision_resampler(image_features, images=image_tensor)
                        
                        # Handle both tuple and tensor returns
                        if isinstance(resampler_output, tuple):
                            resampled_features, aux_loss = resampler_output
                            print(f"DEBUG [Sample {i+1}]: Q-Former Aux Loss: {aux_loss.item():.6f}")
                        else:
                            resampled_features = resampler_output
                        
                        res_sum = resampled_features.sum().item()
                        res_mean = resampled_features.mean().item()
                        res_std = resampled_features.std().item()
                        print(f"DEBUG [Sample {i+1}]: Resampler Sum: {res_sum:.4f}, Mean: {res_mean:.6f}, Std: {res_std:.6f}")
                        
                        # 3. Check Projector Output
                        projected_features = model.get_model().mm_projector(resampled_features)
                        proj_sum = projected_features.sum().item()
                        proj_mean = projected_features.mean().item()
                        proj_std = projected_features.std().item()
                        print(f"DEBUG [Sample {i+1}]: Projector Sum: {proj_sum:.4f}, Mean: {proj_mean:.6f}, Std: {proj_std:.6f}")
                        print(f"DEBUG [Sample {i+1}]: Projector First 5 Values: {projected_features[0, 0, :5].tolist()}")
            except Exception as e:
                print(f"DEBUG: Error checking features: {e}")

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
                
                # Prompt length is text + image tokens; use stored value when using inputs_embeds (e.g. Q-Former)
                prompt_len = getattr(model, "_prompt_length_for_generate", input_ids.shape[1])
                generated_ids = output_ids[0, prompt_len:]
                print(f"DEBUG [Sample {i+1}]: Raw Generated IDs: {generated_ids.tolist()}")

                output = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
                # output = extract_first_sentence(output) # Disable for debugging
                
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
    global NUM_SAMPLES
    
    parser = argparse.ArgumentParser(description="Inference on Q-Former model with auxiliary loss")
    parser.add_argument("--projector", type=str, help="Specific projector to run")
    parser.add_argument("--checkpoint", type=str, help="Direct path to checkpoint directory")
    parser.add_argument("--num_samples", type=int, default=NUM_SAMPLES, help="Number of samples to test")
    args = parser.parse_args()
    
    # Update NUM_SAMPLES if specified
    if args.num_samples:
        NUM_SAMPLES = args.num_samples

    if args.projector:
        # Child process mode
        run_single_inference(args.projector, args.checkpoint)
    else:
        # Parent process mode: spawn a subprocess for each projector
        # This ensures full GPU memory cleanup between runs
        for proj in PROJECTORS:
            print(f"\n{'='*60}")
            print(f"Testing projector: {proj}")
            print(f"{'='*60}\n")
            cmd = [sys.executable, __file__, "--projector", proj, "--num_samples", str(NUM_SAMPLES)]
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"Subprocess for {proj} failed with error: {e}")

if __name__ == "__main__":
    main()