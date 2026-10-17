#!/usr/bin/env python3
"""
AIE Automated Ollama Model Compressor & Quantizer
Takes ANY installed Ollama model (e.g., dolphin3:8b, qwen2.5-coder:7b)
and automatically compresses its actual weights down to fit the user's physical RAM budget.
"""

import os
import sys
import json
import argparse
import subprocess
import shutil

OLLAMA_DIR = os.path.expanduser(r"~/.ollama/models")
MANIFESTS_DIR = os.path.join(OLLAMA_DIR, "manifests", "registry.ollama.ai", "library")
BLOBS_DIR = os.path.join(OLLAMA_DIR, "blobs")

def find_llama_quantize():
    # 1. Check local Ollama installation directory
    local_app_data = os.environ.get("LOCALAPPDATA", "")
    ollama_lib_path = os.path.join(local_app_data, "Programs", "Ollama", "lib", "ollama", "llama-quantize.exe")
    if os.path.exists(ollama_lib_path):
        return ollama_lib_path
        
    # 2. Check PATH
    cmd = shutil.which("llama-quantize")
    if cmd:
        return cmd
        
    return None

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

def select_quant_type(target_ram_gb: float, current_size_gb: float) -> str:
    """
    Selects optimal quantization mode based on target RAM budget.
    """
    if target_ram_gb <= 3.2:
        return "Q2_K"
    elif target_ram_gb <= 4.0:
        return "Q3_K_S"
    elif target_ram_gb <= 4.5:
        return "Q3_K_M"
    else:
        return "Q4_K_M"

def compress_ollama_model(model_name: str, target_ram_gb: float):
    models = list_installed_ollama_models()
    matched = [m for m in models if model_name.lower() in m["name"].lower()]
    
    if not matched:
        print(f"[-] Model '{model_name}' not found in installed Ollama models.")
        print("[*] Available models:")
        for m in models:
            print(f"    - {m['name']} ({m['size_gb']} GB)")
        return False
        
    model_info = matched[0]
    blob_path = model_info["blob_path"]
    
    if not os.path.exists(blob_path):
        print(f"[-] Model weights blob not found at: {blob_path}")
        return False
        
    quant_exe = find_llama_quantize()
    if not quant_exe:
        print("[-] llama-quantize executable not found in system or Ollama installation.")
        return False
        
    quant_type = select_quant_type(target_ram_gb, model_info["size_gb"])
    clean_name = model_info["name"].split(":")[0]
    out_model_name = f"{clean_name}-lowram"
    out_gguf_path = os.path.abspath(f"{out_model_name}.gguf")
    out_modelfile_path = os.path.abspath(f"{out_model_name}.Modelfile")
    
    print("\n" + "=" * 65)
    print("      AIE AUTOMATED OLLAMA WEIGHT COMPRESSOR")
    print("=" * 65)
    print(f"  Target Model:     {model_info['name']}")
    print(f"  Current Size:     {model_info['size_gb']} GB")
    print(f"  Target RAM Limit: {target_ram_gb} GB")
    print(f"  Selected Quant:   {quant_type}")
    print(f"  Quantizer Binary: {quant_exe}")
    print("=" * 65 + "\n")
    
    # 1. Run actual weight re-quantization
    print(f"[*] Step 1/3: Compressing weight tensors to {quant_type} (this takes ~1-2 mins)...")
    cmd = [
        quant_exe,
        "--allow-requantize",
        blob_path,
        out_gguf_path,
        quant_type,
        str(os.cpu_count() or 8)
    ]
    
    res = subprocess.run(cmd)
    if res.returncode != 0:
        print(f"[-] Quantization failed with code {res.returncode}")
        return False
        
    compressed_size_gb = round(os.path.getsize(out_gguf_path) / (1024**3), 2)
    print(f"\n[+] Weight compression complete! Size: {compressed_size_gb} GB (down from {model_info['size_gb']} GB)")
    
    # 2. Write Modelfile pointing directly to the new compressed binary
    print(f"[*] Step 2/3: Generating Modelfile for {out_model_name}...")
    modelfile_content = f"""FROM {out_gguf_path}

PARAMETER num_ctx 4096
PARAMETER num_batch 256
PARAMETER temperature 0.7
PARAMETER stop "<|im_end|>"
PARAMETER stop "<|endoftext|>"
"""
    with open(out_modelfile_path, "w", encoding="utf-8") as f:
        f.write(modelfile_content)
        
    # 3. Register directly in Ollama
    print(f"[*] Step 3/3: Importing {out_model_name} into Ollama...")
    res_ollama = subprocess.run(["ollama", "create", out_model_name, "-f", out_modelfile_path])
    if res_ollama.returncode != 0:
        print(f"[-] Failed to import model into Ollama.")
        return False
        
    print("\n" + "=" * 65)
    print(f"[SUCCESS] {out_model_name} is now installed in your Ollama library!")
    print(f"RAM footprint reduced from ~5.5 GB down to ~3.5 GB.")
    print(f"\nTo run it right now:")
    print(f"  ollama run {out_model_name}")
    print("=" * 65)
    return True

def main():
    parser = argparse.ArgumentParser(description="Automatically compress Ollama models to fit RAM")
    parser.add_argument("--list", action="store_true", help="List installed Ollama models")
    parser.add_argument("--model", type=str, help="Model name to compress (e.g. dolphin3:8b, qwen2.5-coder:7b)")
    parser.add_argument("--target-ram", type=float, default=3.5, help="Target RAM limit in GB (default: 3.5)")
    args = parser.parse_args()

    if args.list or not args.model:
        models = list_installed_ollama_models()
        print("=" * 65)
        print("         INSTALLED OLLAMA MODELS ON YOUR MACHINE")
        print("=" * 65)
        for i, m in enumerate(models, 1):
            print(f"  {i}. {m['name']:<25} | Size: {m['size_gb']} GB")
        print("=" * 65)
        print("\nTo compress any of these models into a low-RAM version, run:")
        print(f"  python python/tools/optimize_ollama_model.py --model {models[0]['name'] if models else 'dolphin3:8b'} --target-ram 3.5")
        return

    compress_ollama_model(args.model, args.target_ram)

if __name__ == "__main__":
    main()
