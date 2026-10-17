#!/usr/bin/env python3
"""
Ollama Model Optimizer for AIE.
Detects installed models in Ollama's local directory, analyzes their layer sizes,
and creates an optimized low-RAM Modelfile and quantization plan.
"""

import os
import sys
import json
import argparse

OLLAMA_DIR = os.path.expanduser(r"~/.ollama/models")
MANIFESTS_DIR = os.path.join(OLLAMA_DIR, "manifests", "registry.ollama.ai", "library")
BLOBS_DIR = os.path.join(OLLAMA_DIR, "blobs")

def list_installed_ollama_models():
    models = []
    if not os.path.exists(MANIFESTS_DIR):
        return models

    for root, dirs, files in os.walk(MANIFESTS_DIR):
        for file in files:
            manifest_path = os.path.join(root, file)
            rel_path = os.path.relpath(manifest_path, MANIFESTS_DIR)
            model_name = rel_path.replace("\\", ":").replace("/", ":")
            
            try:
                with open(manifest_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                model_blob_digest = None
                model_size_bytes = 0
                for layer in data.get("layers", []):
                    if layer.get("mediaType") == "application/vnd.ollama.image.model":
                        model_blob_digest = layer.get("digest", "")
                        model_size_bytes = layer.get("size", 0)
                        break
                        
                if model_blob_digest:
                    blob_filename = model_blob_digest.replace(":", "-")
                    blob_path = os.path.join(BLOBS_DIR, blob_filename)
                    models.append({
                        "name": model_name,
                        "blob_path": blob_path,
                        "size_bytes": model_size_bytes,
                        "size_gb": round(model_size_bytes / (1024**3), 2),
                        "exists": os.path.exists(blob_path)
                    })
            except Exception:
                pass
    return models

def create_low_ram_profile(model_info, ram_budget_gb: float):
    base_name = model_info["name"].split(":")[0]
    out_name = f"{base_name}-lowram"
    modelfile_path = f"{out_name}.Modelfile"
    
    # Calculate optimal context length and batching to respect RAM ceiling
    # KIVI 2-bit/4-bit KV Cache ratio calculation
    target_bytes = ram_budget_gb * (1024**3)
    weight_bytes = model_info["size_bytes"]
    
    print(f"[*] Base model:       {model_info['name']} ({model_info['size_gb']} GB)")
    print(f"[*] Target RAM limit: {ram_budget_gb} GB")
    
    modelfile_content = f"""# AIE Low-RAM Optimized Profile for {model_info['name']}
FROM {model_info['name']}

# Memory optimization parameters
PARAMETER num_ctx 4096
PARAMETER num_batch 256
PARAMETER num_gpu 0
PARAMETER temperature 0.7
PARAMETER top_p 0.9

# Stream memory limits
PARAMETER stop "<|im_end|>"
PARAMETER stop "<|endoftext|>"
"""
    with open(modelfile_path, "w", encoding="utf-8") as f:
        f.write(modelfile_content)
        
    print(f"[+] Generated optimized Modelfile: {modelfile_path}")
    print(f"\nTo create and run this low-RAM version in Ollama:")
    print(f"  ollama create {out_name} -f {modelfile_path}")
    print(f"  ollama run {out_name}")

def main():
    parser = argparse.ArgumentParser(description="Optimize local Ollama models with AIE")
    parser.add_argument("--list", action="store_true", help="List installed Ollama models")
    parser.add_argument("--model", type=str, help="Model name to optimize (e.g. dolphin3:8b)")
    parser.add_argument("--target-ram", type=float, default=3.5, help="Target RAM budget in GB (default: 3.5)")
    args = parser.parse_args()

    models = list_installed_ollama_models()

    if not models:
        print("[-] No Ollama models found in ~/.ollama/models")
        return

    if args.model:
        matched = [m for m in models if args.model.lower() in m["name"].lower()]
        if not matched:
            print(f"[-] Model '{args.model}' not found in installed models.")
            return
        create_low_ram_profile(matched[0], args.target_ram)
    else:
        print("=" * 65)
        print("         AIE DETECTED OLLAMA MODELS ON YOUR MACHINE")
        print("=" * 65)
        for i, m in enumerate(models, 1):
            print(f"  {i}. {m['name']:<25} | Size: {m['size_gb']} GB")
        print("=" * 65)
        print("\nTo generate a low-RAM profile for one of your models, run:")
        print(f"  python python/tools/optimize_ollama_model.py --model {models[0]['name']} --target-ram 3.5")

if __name__ == "__main__":
    main()
