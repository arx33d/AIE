#!/usr/bin/env python3
"""
AIE MILP Pareto Knapsack Solver
Solves:
   min   Sum_{l} Sum_{b} x_{l,b} * (S_l * Distortion(b))
   s.t.  Sum_{l} Sum_{b} x_{l,b} * SizeBytes(l, b) + KV_Cache + Buffers <= M_budget
         Sum_{b} x_{l,b} == 1 for all l
"""

import sys
import json
import numpy as np

try:
    from scipy.optimize import milp, LinearConstraint, Bounds
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

DISTORTION_MAP = {
    16: 0.00000,
    8:  0.00085,
    4:  0.01250,
    3:  0.05400,
    2:  0.21500
}

BYTES_PER_ELEMENT = {
    16: 2.0,
    8:  1.0,
    4:  0.5,
    3:  0.375,
    2:  0.25
}

def solve_knapsack(sensitivity_json_path: str, ram_budget_bytes: int, context_tokens: int = 8192) -> dict:
    with open(sensitivity_json_path, 'r') as f:
        data = json.load(f)
        
    tensors = list(data.items())
    num_tensors = len(tensors)
    precisions = [2, 3, 4, 8, 16]
    num_precisions = len(precisions)
    
    # Pre-allocate KIVI Paged KV Cache memory footprint (2-bit K, 4-bit V per head)
    # 32 layers, 8 KV heads, 128 head_dim = 2 * 32 * 8 * 128 * (0.25 + 0.5) bytes/token
    kv_cache_bytes = int(context_tokens * 32 * 8 * 128 * 0.75)
    working_buffers_bytes = 256 * 1024 * 1024 # 256 MB scratchpad
    
    net_weight_budget = ram_budget_bytes - (kv_cache_bytes + working_buffers_bytes)
    
    print(f"[*] Total Physical RAM Budget: {ram_budget_bytes / 1024**3:.2f} GB")
    print(f"[*] KV Cache Allocation:     {kv_cache_bytes / 1024**2:.2f} MB")
    print(f"[*] Scratchpad Buffers:      {working_buffers_bytes / 1024**2:.2f} MB")
    print(f"[*] Net Weight Budget:       {net_weight_budget / 1024**3:.2f} GB")
    
    plan = {
        "budget_bytes": ram_budget_bytes,
        "kv_cache_bytes": kv_cache_bytes,
        "fits_in_ram": True,
        "allocations": {},
        "out_of_core_streaming_layers": []
    }
    
    if HAS_SCIPY and net_weight_budget > 0:
        c = []
        size_coefficients = []
        
        for _, meta in tensors:
            s_l = meta["sensitivity"]
            numel = meta["numel"]
            is_protected = meta.get("protected", False)
            
            for p in precisions:
                if is_protected and p < 4:
                    cost = s_l * 1e6
                else:
                    cost = s_l * DISTORTION_MAP[p]
                c.append(cost)
                size_coefficients.append(numel * BYTES_PER_ELEMENT[p])
                
        c = np.array(c)
        A_budget = np.array([size_coefficients])
        
        A_eq = np.zeros((num_tensors, num_tensors * num_precisions))
        for i in range(num_tensors):
            for j in range(num_precisions):
                A_eq[i, i * num_precisions + j] = 1.0
                
        A_total = np.vstack([A_budget, A_eq])
        
        lhs = np.zeros(1 + num_tensors)
        rhs = np.zeros(1 + num_tensors)
        
        lhs[0] = 0.0
        rhs[0] = float(net_weight_budget)
        
        for i in range(num_tensors):
            lhs[1 + i] = 1.0
            rhs[1 + i] = 1.0
            
        constraints = LinearConstraint(A_total, lhs, rhs)
        integrality = np.ones(num_tensors * num_precisions)
        bounds = Bounds(0.0, 1.0)
        
        res = milp(c=c, integrality=integrality, constraints=constraints, bounds=bounds)
        
        if res.success:
            print("[+] Optimal Pareto Quantization Plan found via HiGHS MILP.")
            x_sol = res.x.round().astype(int)
            for i, (name, meta) in enumerate(tensors):
                for j, p in enumerate(precisions):
                    if x_sol[i * num_precisions + j] == 1:
                        plan["allocations"][name] = {
                            "bits": p,
                            "location": "pinned_ram",
                            "size_bytes": int(meta["numel"] * BYTES_PER_ELEMENT[p])
                        }
            return plan

    # Deterministic dynamic greedy knapsack solver fallback
    print("[*] Running Pareto greedy allocation with out-of-core fallback...")
    accumulated_ram = 0
    for name, meta in tensors:
        numel = meta["numel"]
        is_protected = meta.get("protected", False)
        
        assigned_bits = 8 if is_protected else 2
        alloc_size = int(numel * BYTES_PER_ELEMENT[assigned_bits])
        
        if accumulated_ram + alloc_size <= net_weight_budget:
            plan["allocations"][name] = {
                "bits": assigned_bits,
                "location": "pinned_ram",
                "size_bytes": alloc_size
            }
            accumulated_ram += alloc_size
        else:
            plan["fits_in_ram"] = False
            plan["allocations"][name] = {
                "bits": assigned_bits,
                "location": "nvme_stream",
                "size_bytes": alloc_size
            }
            plan["out_of_core_streaming_layers"].append(name)

    return plan

def main():
    if len(sys.argv) < 4:
        print(f"Usage: {sys.argv[0]} <sensitivity.json> <ram_budget_gb> <out_plan.json>")
        sys.exit(1)
        
    sens_path = sys.argv[1]
    ram_gb = float(sys.argv[2])
    out_path = sys.argv[3]
    
    budget_bytes = int(ram_gb * (1024**3))
    plan = solve_knapsack(sens_path, budget_bytes)
    
    with open(out_path, 'w') as f:
        json.dump(plan, f, indent=2)
    print(f"[+] Quantization plan written to: {out_path}")

if __name__ == "__main__":
    main()
