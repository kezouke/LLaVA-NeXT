import os
import json
import argparse
import subprocess
import sys
import torch
from PIL import Image
from tqdm import tqdm
import math

# Add parent directory to path to allow importing llava modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates

# Default Configurations
DEFAULT_CHECKPOINT_ROOT = "./checkpoints"
DEFAULT_PROJECTORS = [
    "linear",
    "mlp2x",
    "mlp2x-res2x",
    "pooler",
    "mlp_1layer_8m",
    "mlp_1layer_12m",
    "mlp_2x_narrow_15m"
]

def get_checkpoint_path(checkpoint_root, projector_slug, tag):
    if tag:
        model_dir_name = f"llava-{projector_slug}-{tag}"
    else:
        model_dir_name = f"llava-{projector_slug}"
    model_dir = os.path.join(checkpoint_root, model_dir_name)
    
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

def load_coco_data(image_folder, annotation_file=None):
    """
    Load COCO data. 
    If annotation_file is provided, load image_ids from it.
    Otherwise, list images in image_folder.
    """
    images = []
    
    if annotation_file and os.path.exists(annotation_file):
        print(f"Loading annotations from {annotation_file}")
        with open(annotation_file, 'r') as f:
            data = json.load(f)
        
        # Handle standard COCO format
        if 'images' in data:
            for img in data['images']:
                images.append({
                    'file_name': img['file_name'],
                    'id': img['id']
                })
        else:
            # Handle list format if applicable
            pass
    else:
        print(f"Listing images from {image_folder}")
        for f in os.listdir(image_folder):
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                # Try to extract ID from filename (e.g., COCO_val2014_000000000042.jpg)
                try:
                    img_id = int(f.split('_')[-1].split('.')[0])
                except ValueError:
                    img_id = f # Fallback
                
                images.append({
                    'file_name': f,
                    'id': img_id
                })
    
    # Sort by ID for consistency
    images.sort(key=lambda x: x['id'] if isinstance(x['id'], int) else x['file_name'])
    return images

def extract_caption(text):
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

def evaluate_coco(result_file, annotation_file):
    from pycocotools.coco import COCO
    from pycocoevalcap.eval import COCOEvalCap

    coco = COCO(annotation_file)
    coco_result = coco.loadRes(result_file)

    # create coco_eval object by taking coco and coco_result
    coco_eval = COCOEvalCap(coco, coco_result)

    # evaluate on full split (restriction removed per original comment)
    # coco_eval.params['image_id'] = coco_result.getImgIds()

    # evaluate results
    # SPICE can take a while, so maybe disable it for quick checks if needed
    # coco_eval.evaluate()
    
    # For now, let's run standard evaluation
    coco_eval.evaluate()

    # print output evaluation scores
    print("\nEvaluation Scores:")
    for metric, score in coco_eval.eval.items():
        print(f"{metric}: {score:.3f}")
    
    return coco_eval.eval

def run_inference(args, projector):
    print(f"\n{'='*40}")
    print(f"Processing Projector: {projector}")
    print(f"{'='*40}")

    ckpt_path = get_checkpoint_path(args.checkpoint_root, projector, args.train_tag)
    if not ckpt_path:
        print(f"Skipping {projector} due to missing checkpoint.")
        return

    print(f"Loading checkpoint: {ckpt_path}")
    
    try:
        model_name = get_model_name_from_path(ckpt_path)
        if "llama" not in model_name.lower():
            model_name = model_name + "-llama"
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        print(f"Using dtype: {dtype}")

        tokenizer, model, image_processor, context_len = load_pretrained_model(
            model_path=ckpt_path,
            model_base=args.base_model_path,
            model_name=model_name,
            device_map=device,
            torch_dtype=dtype,
            multimodal=True
        )
        
        model.to(dtype=dtype)

        # Manually reload projector weights (fix for some LLaVA versions/setups)
        print("Manually reloading projector weights...")
        mm_projector_path = os.path.join(ckpt_path, "mm_projector.bin")
        if os.path.exists(mm_projector_path):
            mm_projector_weights = torch.load(mm_projector_path, map_location="cpu")
            mm_projector_weights = {k: v.to(dtype) for k, v in mm_projector_weights.items()}
            missing, unexpected = model.load_state_dict(mm_projector_weights, strict=False, assign=True)
            print(f"Reloaded projector. Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
            model.get_model().mm_projector.to(device)
            if hasattr(model.get_model(), 'vision_resampler'):
                model.get_model().vision_resampler.to(device)
        else:
            print(f"Warning: mm_projector.bin not found at {mm_projector_path}")

        # Load Data
        images = load_coco_data(args.image_folder, args.annotation_file)
        print(f"Found {len(images)} images.")
        
        if args.max_samples:
            images = images[:args.max_samples]
            print(f"Limiting to {args.max_samples} samples.")

        results = []
        
        # Prepare prompt -- use conversation template if specified
        conv_template = getattr(args, "conv_template", None)
        if conv_template and conv_template in conv_templates:
            conv = conv_templates[conv_template].copy()
            conv.tokenizer = tokenizer
            conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\nPlease describe this image in one sentence.")
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
        else:
            prompt = DEFAULT_IMAGE_TOKEN + "\nPlease describe this image in one sentence."
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(device)

        for img_info in tqdm(images, desc="Inference"):
            image_file = img_info['file_name']
            image_id = img_info['id']
            image_path = os.path.join(args.image_folder, image_file)
            
            if not os.path.exists(image_path):
                continue
                
            try:
                image = Image.open(image_path).convert('RGB')
                image_tensor = process_images([image], image_processor, model.config)
                
                if type(image_tensor) is list:
                    image_tensor = [img.to(device, dtype=dtype) for img in image_tensor]
                else:
                    image_tensor = image_tensor.to(device, dtype=dtype)

                with torch.inference_mode():
                    output_ids = model.generate(
                        input_ids,
                        images=image_tensor,
                        image_sizes=[image.size],
                        do_sample=False, # Greedy decoding for reproducibility
                        max_new_tokens=64,
                        use_cache=True,
                        pad_token_id=tokenizer.eos_token_id
                    )

                # Use _prompt_length_for_generate if available (handles image token expansion)
                prompt_len = getattr(model, "_prompt_length_for_generate", input_ids.shape[1])
                output = tokenizer.decode(output_ids[0, prompt_len:], skip_special_tokens=True).strip()
                caption = extract_caption(output)
                
                results.append({
                    "image_id": image_id,
                    "caption": caption
                })
                
                # Optional: Print first few results for sanity check
                if len(results) <= 5:
                    print(f"\n[Sample {len(results)}] Image: {image_file}")
                    print(f"Prediction: {caption}")

            except Exception as e:
                print(f"Error processing {image_file}: {e}")
                continue

        # Save results
        output_dir = args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        result_file = os.path.join(output_dir, f"captions_{args.split}_{projector}_results.json")
        
        with open(result_file, 'w') as f:
            json.dump(results, f)
        
        print(f"Results saved to {result_file}")

        # Evaluate if annotations are available
        if args.do_eval and args.annotation_file:
            try:
                eval_results = evaluate_coco(result_file, args.annotation_file)
                
                # Save evaluation results to a file
                eval_file = os.path.join(output_dir, f"coco_metrics_{args.split}_{projector}.json")
                with open(eval_file, 'w') as f:
                    json.dump(eval_results, f, indent=4)
                print(f"Evaluation metrics saved to {eval_file}")
                
            except Exception as e:
                print(f"Evaluation failed: {e}")

    except Exception as e:
        print(f"Error running model {projector}: {e}")
        import traceback
        traceback.print_exc()

def main():
    parser = argparse.ArgumentParser(description="Evaluate LLaVA Projector Ablations on COCO")
    parser.add_argument("--base_model_path", type=str, default="/data/ssd_storage/llama/", help="Path to base LLM")
    parser.add_argument("--checkpoint_root", type=str, default=DEFAULT_CHECKPOINT_ROOT, help="Root directory of checkpoints")
    parser.add_argument("--image_folder", type=str, required=True, help="Path to image folder (e.g., val2014)")
    parser.add_argument("--annotation_file", type=str, help="Path to COCO annotation file (e.g., captions_val2014.json)")
    parser.add_argument("--train_tag", type=str, default="train2014", help="Tag used in training output directory name")
    parser.add_argument("--split", type=str, default="val2014", help="Split name (val2014, test2014)")
    parser.add_argument("--projectors", nargs="+", default=DEFAULT_PROJECTORS, help="List of projectors to evaluate")
    parser.add_argument("--output_dir", type=str, default="./results", help="Directory to save results")
    parser.add_argument("--max_samples", type=int, help="Limit number of samples for debugging")
    parser.add_argument("--do_eval", action="store_true", help="Run evaluation after inference (requires annotation_file)")
    parser.add_argument("--conv_template", type=str, default=None, help="Conversation template (e.g., llava_llama_3). If not set, uses plain prompt.")
    
    # Internal argument for subprocess
    parser.add_argument("--run_projector", type=str, help="Internal: run specific projector")
    
    args = parser.parse_args()

    if args.run_projector:
        # Child process
        run_inference(args, args.run_projector)
    else:
        # Parent process
        for proj in args.projectors:
            print(f"Launching subprocess for {proj}...")
            
            # Construct command
            cmd = [sys.executable, __file__]
            
            # Pass through all arguments
            cmd.extend(["--base_model_path", args.base_model_path])
            cmd.extend(["--checkpoint_root", args.checkpoint_root])
            cmd.extend(["--image_folder", args.image_folder])
            if args.annotation_file:
                cmd.extend(["--annotation_file", args.annotation_file])
            cmd.extend(["--train_tag", args.train_tag])
            cmd.extend(["--split", args.split])
            cmd.extend(["--output_dir", args.output_dir])
            if args.max_samples:
                cmd.extend(["--max_samples", str(args.max_samples)])
            if args.do_eval:
                cmd.append("--do_eval")
            if args.conv_template:
                cmd.extend(["--conv_template", args.conv_template])
            
            # Add the specific projector to run
            cmd.extend(["--run_projector", proj])
            
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"Subprocess for {proj} failed with error: {e}")

if __name__ == "__main__":
    main()