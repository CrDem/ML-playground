import sys
import os
import torch
from safetensors import safe_open
from pathlib import Path


if (len(sys.argv) != 3):
    print("Usage: python3 ./check.py <input> <output>")
    sys.exit(1)



SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_DIR = SCRIPT_DIR.parent
INPUT_PATH = str(PROJECT_DIR / sys.argv[1])
OUTPUT_PATH = str(PROJECT_DIR / sys.argv[2])
print("input: ", INPUT_PATH)
print("output: ", OUTPUT_PATH)

with safe_open(INPUT_PATH, framework="pt") as f:
    expected_result = f.get_tensor("C")

with safe_open(OUTPUT_PATH, framework="pt") as f:
    gemm_result = f.get_tensor("C")

mse = torch.mean((gemm_result.double() - expected_result.double()) ** 2)
rmse = torch.sqrt(mse)

print("GeMM Metal result:", gemm_result, sep="\n")

print("Torch result:", expected_result, sep="\n")

print(f"RMSE: {rmse.item():.12f}")
