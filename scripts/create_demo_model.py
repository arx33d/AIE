#!/usr/bin/env python3
"""
Creates a compact, valid .safetensors model file for instant end-to-end testing of AIE.
"""

import struct
import json
import os

def create_demo_model(output_path: str = "demo_model.safetensors"):
    # Define a minimal valid Transformer topology (2 layers, 512 hidden_size, 4 heads, 8192 vocab)
    header = {
        "model.embed_tokens.weight": {
            "dtype": "F32",
            "shape": [8192, 512],
            "data_offsets": [0, 16777216]
        },
        "model.layers.0.self_attn.q_proj.weight": {
            "dtype": "F16",
            "shape": [512, 512],
            "data_offsets": [16777216, 17301504]
        },
        "model.layers.0.self_attn.k_proj.weight": {
            "dtype": "F16",
            "shape": [512, 512],
            "data_offsets": [17301504, 17825792]
        },
        "model.layers.0.self_attn.v_proj.weight": {
            "dtype": "F16",
            "shape": [512, 512],
            "data_offsets": [17825792, 18350080]
        },
        "model.layers.0.mlp.gate_proj.weight": {
            "dtype": "F16",
            "shape": [1376, 512],
            "data_offsets": [18350080, 19759104]
        },
        "model.layers.0.mlp.down_proj.weight": {
            "dtype": "F16",
            "shape": [512, 1376],
            "data_offsets": [19759104, 21168128]
        },
        "model.layers.1.self_attn.q_proj.weight": {
            "dtype": "F16",
            "shape": [512, 512],
            "data_offsets": [21168128, 21692416]
        },
        "model.layers.1.mlp.gate_proj.weight": {
            "dtype": "F16",
            "shape": [1376, 512],
            "data_offsets": [21692416, 23101440]
        },
        "lm_head.weight": {
            "dtype": "F32",
            "shape": [8192, 512],
            "data_offsets": [23101440, 39878656]
        }
    }
    
    header_json = json.dumps(header).encode("utf-8")
    header_len = len(header_json)
    
    print(f"[*] Creating demo safetensors file: {output_path}")
    with open(output_path, "wb") as f:
        # Write 8-byte little endian header length
        f.write(struct.pack("<Q", header_len))
        f.write(header_json)
        # Write a small buffer to back data offsets
        total_data_bytes = 39878656
        # Write sparse dummy data (or seek to total size)
        f.seek(8 + header_len + total_data_bytes - 1)
        f.write(b"\x00")
        
    print(f"[+] Successfully generated demo model ({os.path.getsize(output_path) / (1024*1024):.1f} MB)")

if __name__ == "__main__":
    create_demo_model()
