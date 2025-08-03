import sys
import os
import torch
import numpy as np
from safetensors.torch import save_file
from safetensors import safe_open
from pathlib import Path


# MxK * KxN = MxN

if (len(sys.argv) != 4):
    print("Usage: python3 ./generate_data.py <M> <K> <N> (MxK * KxN = MxN)")
    sys.exit(1)

SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_DIR = SCRIPT_DIR.parent
DATA_DIR = PROJECT_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
GEN_DATA_FILE = DATA_DIR / "matmul_data.safetensors"
RESULT_FILE = DATA_DIR / "output.safetensors"



M = int(sys.argv[1])
K = int(sys.argv[2])
N = int(sys.argv[3])

A = torch.randn(M, K, dtype=torch.float16)
B = torch.randn(K, N, dtype=torch.float16)
C = torch.matmul(A.float(), B.float()).half()

save_file({
    "A": A,
    "B": B,
    "C": C
}, str(GEN_DATA_FILE))

print("input file created: ./data/matmul_data.safetensors")

import subprocess

rust_bin = str(PROJECT_DIR) + "/target/release/volgaray_gemm"
env = os.environ.copy()
env["RUST_LOG"]="info"

with subprocess.Popen(
    [rust_bin, "-i", GEN_DATA_FILE, "-o", RESULT_FILE],
    env=env,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    bufsize=1
) as process:
    for line in process.stdout:
        print(line.strip())
    for line in process.stderr:
        print(line.strip(), file=sys.stderr)


with safe_open(RESULT_FILE, framework="pt") as f:
    gemm_result = f.get_tensor("C")



mse = torch.mean((gemm_result.double() - C.double()) ** 2)
rmse = torch.sqrt(mse)

print("Our result:", gemm_result, sep="\n")

print("Torch result:", C, sep="\n")

print(f"RMSE: {rmse.item():.12f}")
