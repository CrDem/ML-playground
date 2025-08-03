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

M = int(sys.argv[1])
K = int(sys.argv[2])
N = int(sys.argv[3])

A = torch.randn(M,K)
B = torch.randn(K,N)
C = torch.matmul(A,B)

save_file({
    "A": A,
    "B": B,
    "C": C
}, str(GEN_DATA_FILE))
