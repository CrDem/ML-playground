### Info
GeMM: MxK @ KxN = MxN  
**!** shaders/matmul.cu need to be pre-compiled

### Usage:
```bash
cargo build -r
python3 ./scripts/generate_and_run.py <M> <N> <K>
```

### Example:
```cmd
(.venv) D:\work\GEMM>python ./scripts/generate_and_run.py 1024 1024 1024
input file created: ./data/matmul_data.safetensors
Computation took 1.878208041191101 ms
GFLOPS: 1142.8100747768055
[2025-08-10T15:16:05Z INFO  volgaray_gemm] Reading tensors from file
[2025-08-10T15:16:05Z INFO  volgaray_gemm] Parsing tensors to matrixes
[2025-08-10T15:16:05Z INFO  volgaray_gemm] Initialization cuda context
[2025-08-10T15:16:05Z INFO  volgaray_gemm] Binding data
[2025-08-10T15:16:05Z INFO  volgaray_gemm] Loading PTX module
[2025-08-10T15:16:05Z INFO  volgaray_gemm] Launching kernel
[2025-08-10T15:16:05Z INFO  volgaray_gemm] Copying result back
[2025-08-10T15:16:05Z INFO  volgaray_gemm] Saving results
Our result:
tensor([[ 45.1875,  11.8359,   6.5312,  ...,   9.1406, -39.2812, -37.8125],
        [ 28.4375,  -8.7812,  22.7656,  ...,  15.0078,  -5.0469,  22.6719],
        [-26.2031, -36.3750,  -8.2266,  ..., -17.6406, -35.2500,  -8.7031],
        ...,
        [-46.1250, -14.7344,  11.0547,  ...,  43.6250,  21.9062,  -1.7305],
        [ 28.8125,  75.8750,  21.5000,  ..., -49.6875, -30.1094,  12.8203],
        [-24.2812,  10.1328, -25.7344,  ...,  -6.6016,  15.7422,  20.9219]],
       dtype=torch.float16)
Torch result:
tensor([[ 45.1875,  11.8359,   6.5312,  ...,   9.1406, -39.2812, -37.8125],
        [ 28.4375,  -8.7812,  22.7656,  ...,  15.0078,  -5.0469,  22.6719],
        [-26.2031, -36.3750,  -8.2266,  ..., -17.6406, -35.2500,  -8.7031],
        ...,
        [-46.1250, -14.7344,  11.0547,  ...,  43.6250,  21.9062,  -1.7305],
        [ 28.8125,  75.8750,  21.5000,  ..., -49.6875, -30.1094,  12.8203],
        [-24.2812,  10.1328, -25.7344,  ...,  -6.6016,  15.7422,  20.9219]],
       dtype=torch.float16)
RMSE: 0.000574388377
```

### Benchmarks:
Device: NVIDIA GeForce RTX 2070 SUPER

Kernel: Tiled Matmul - Half Precision with Float Accumulation  
Tile size: 32x32
| M    | K    | N    | GFLOPS  | RMSE     |
| ---- | ---- | ---- | ------- | -------- |
| 256  | 256  | 256  | 628.92  | 0.000165 |
| 512  | 512  | 512  | 1007.99 | 0.000329 |
| 1024 | 1024 | 1024 | 1142.81 | 0.000574 |
| 2048 | 2048 | 2048 | 274.84  | 0.000954 |
| 512  | 1024 | 2048 | 1107.58 | 0.000550 |

Kernel: The simplest tensor cores matmul
Tile size: 16x16 (processed by single warp)
| M    | K    | N    | GFLOPS  | RMSE     |
| ---- | ---- | ---- | ------- | -------- |
| 256  | 256  | 256  | 1168.00 | 0.000227 |
| 512  | 512  | 512  | 2788.82 | 0.000540 |
| 1024 | 1024 | 1024 | 3617.91 | 0.001059 |
| 2048 | 2048 | 2048 |  927.90 | 0.002071 |
| 512  | 1024 | 2048 | 3610.51 | 0.001053 |