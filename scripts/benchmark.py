import subprocess
import re
import os
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent.resolve()
GEN_SCRIPT = PROJECT_DIR / "scripts" / "generate_and_run.py"

ENV = os.environ.copy()
ENV["RUST_LOG"] = "info"

BENCHMARKS = [
    (256, 256, 256),
    (512, 512, 512),
    (1024, 1024, 1024),
    (2048, 2048, 2048),
    (512, 1024, 2048),
]

GFLOPS_RE = re.compile(r"GFLOPS:\s*([0-9.]+)")
RMSE_RE = re.compile(r"RMSE:\s*([0-9.eE+-]+)")

results = []

print("Running benchmarks...\n")

for M, K, N in BENCHMARKS:
    print(f"→ M={M}, K={K}, N={N}")

    proc = subprocess.run(
        [sys.executable, GEN_SCRIPT, str(M), str(K), str(N)],
        cwd=PROJECT_DIR,
        env=ENV,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    output = proc.stdout + "\n" + proc.stderr
    print(output)

    gflops_match = GFLOPS_RE.search(output)
    rmse_match = RMSE_RE.search(output)

    if not gflops_match or not rmse_match:
        raise RuntimeError(
            f"Failed to parse results for M={M}, K={K}, N={N}\n\n{output}"
        )

    gflops = float(gflops_match.group(1))
    rmse = float(rmse_match.group(1))

    results.append((M, K, N, gflops, rmse))

# ---------------------------
# Markdown-таблица
# ---------------------------
print("\n### Benchmarks:")
print("Device: NVIDIA GeForce RTX 2070 SUPER\n")
print("Kernel: Tiled Matmul - Half Precision with Float Accumulation  ")
print("Tile size: 32x32\n")

print("| M    | K    | N    | GFLOPS  | RMSE     |")
print("| ---- | ---- | ---- | ------- | -------- |")

for M, K, N, gflops, rmse in results:
    print(f"| {M:<4} | {K:<4} | {N:<4} | {gflops:7.2f} | {rmse:.6f} |")