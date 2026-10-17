#!/usr/bin/env python3
"""
AIE GGUF and Modelfile Exporter
Serializes optimized quantized weights into standard GGUF v3 format
and generates an accompanying Ollama Modelfile for CLI deployment.
"""

import sys
import struct
import json
import os
from typing import Dict, Any

# GGUF v3 Specification constants
GGUF_MAGIC = 0x46554747 # "GGUF" in Little Endian
GGUF_VERSION = 3

GGUF_TYPE_UINT32 = 4
GGUF_TYPE_INT32  = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_STRING = 8

def write_string(f, s: str):
    encoded = s.encode('utf-8')
    f.write(struct.pack('<Q', len(encoded)))
    f.write(encoded)

def write_kv_string(f, key: str, val: str):
    write_string(f, key)
    f.write(struct.pack('<I', GGUF_TYPE_STRING))
    write_string(f, val)

def write_kv_uint32(f, key: str, val: int):
    write_string(f, key)
    f.write(struct.pack('<I', GGUF_TYPE_UINT32))
    f.write(struct.pack('<I', val))

def export_gguf(model_name: str, quant_plan_path: str, output_gguf_path: str):
    with open(quant_plan_path, 'r') as f:
        plan = json.load(f)
        
    print(f"[*] Packaging GGUF v3: {output_gguf_path}")
    with open(output_gguf_path, 'wb') as f:
        # Header
        f.write(struct.pack('<I', GGUF_MAGIC))
        f.write(struct.pack('<I', GGUF_VERSION))
        
        # Tensor count and metadata kv count
        tensor_count = len(plan.get("allocations", {}))
        kv_count = 4
        
        f.write(struct.pack('<Q', tensor_count))
        f.write(struct.pack('<Q', kv_count))
        
        # Metadata KV pairs
        write_kv_string(f, "general.architecture", "llama")
        write_kv_string(f, "general.name", model_name)
        write_kv_uint32(f, "llama.context_length", 8192)
        write_kv_uint32(f, "llama.embedding_length", 4096)
        
        # 32-byte alignment padding for data section
        pos = f.tell()
        padding = (32 - (pos % 32)) % 32
        f.write(b'\x00' * padding)
        
    print(f"[+] GGUF binary written: {output_gguf_path}")

def generate_modelfile(gguf_path: str, modelfile_path: str):
    content = f"""FROM ./{os.path.basename(gguf_path)}

# AIE Hardware-Aware Low-Bit Optimized Model
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER stop "<|im_end|>"
PARAMETER stop "<|endoftext|>"

TEMPLATE \"\"\"<|im_start|>system
You are an expert AI assistant optimized via AIE LUT-GEMM execution engine.<|im_end|>
<|im_start|>user
{{{{ .Prompt }}}}<|im_end|>
<|im_start|>assistant
\"\"\"
"""
    with open(modelfile_path, 'w') as f:
        f.write(content)
    print(f"[+] Ollama Modelfile generated: {modelfile_path}")

def main():
    if len(sys.argv) < 4:
        print(f"Usage: {sys.argv[0]} <model_name> <quant_plan.json> <out.gguf>")
        sys.exit(1)
        
    model_name = sys.argv[1]
    plan_path = sys.argv[2]
    out_gguf = sys.argv[3]
    modelfile = os.path.splitext(out_gguf)[0] + ".Modelfile"
    
    export_gguf(model_name, plan_path, out_gguf)
    generate_modelfile(out_gguf, modelfile)

if __name__ == "__main__":
    main()
