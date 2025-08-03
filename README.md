### Info
GeMM: MxK @ KxN = MxN
! shaders/matmul.cu need to be pre-compiled

### Usage:
```bash
cargo build -r
python3 ./scripts/generate_and_run.py <M> <N> <K>
```

### Example:
```cmd
(.venv) D:\work\GEMM>python ./scripts/generate_and_run.py 1024 1024 1024
input file created: ./data/matmul_data.safetensors
Computation took 2 ms
GFLOPS: 1058.765388447689
[2025-08-03T21:15:37Z INFO  volgaray_gemm] Reading tensors from file
[2025-08-03T21:15:37Z INFO  volgaray_gemm] Parsing tensors to matrixes
[2025-08-03T21:15:37Z INFO  volgaray_gemm] Initialization cuda context
[2025-08-03T21:15:37Z INFO  volgaray_gemm] Binding data
[2025-08-03T21:15:37Z INFO  volgaray_gemm] Loading PTX module
[2025-08-03T21:15:37Z INFO  volgaray_gemm] Launching kernel
[2025-08-03T21:15:37Z INFO  volgaray_gemm] Copying result back
[2025-08-03T21:15:37Z INFO  volgaray_gemm] Saving results
Our result:
tensor([[-14.6562,  53.3125, -22.1406,  ..., -11.0703, -24.4531,  23.2031],
        [ -6.4062,   1.8330,  -2.9766,  ...,   6.5312,  57.4688,  37.0938],
        [ 21.0312, -11.8047, -38.0312,  ...,  61.6875,  27.6719,  13.0547],
        ...,
        [ 43.2500, -29.7500,  40.4375,  ...,  -3.2383,  33.9375,   1.4277],
        [ 36.5938,  -5.3867,   7.7422,  ...,  19.2188,  51.0938, -15.0234],
        [-10.1797,   4.6602,  -7.7344,  ..., -39.1875,  -9.5156,  -4.8477]],
       dtype=torch.float16)
Torch result:
tensor([[-14.6562,  53.3125, -22.1406,  ..., -11.0703, -24.4531,  23.2031],
        [ -6.4062,   1.8330,  -2.9766,  ...,   6.5312,  57.4688,  37.0938],
        [ 21.0312, -11.8047, -38.0312,  ...,  61.6875,  27.6719,  13.0547],
        ...,
        [ 43.2500, -29.7500,  40.4375,  ...,  -3.2383,  33.9375,   1.4277],
        [ 36.5938,  -5.3867,   7.7422,  ...,  19.2188,  51.0938, -15.0234],
        [-10.1797,   4.6602,  -7.7344,  ..., -39.1875,  -9.5156,  -4.8477]],
       dtype=torch.float16)
RMSE: 0.000583171239
```

### benchmarks:
Device: NVIDIA GeForce RTX 2070 SUPER

Kernel: Tiled Matmul - Half Precision with Float Accumulation
Tile size: 32x32
| M    | K    | N    | GFLOPS  | RMSE     |
| ---- | ---- | ---- | ------- | -------- |
| 256  | 256  | 256  | 174.79  | 0.000198 |
| 512  | 512  | 512  | 593.04  | 0.000331 |
| 1024 | 1024 | 1024 | 1058.77 | 0.000583 |
| 2048 | 2048 | 2048 | 266.86  | 0.000985 |
| 512  | 1024 | 2048 | 1000.20 | 0.000615 |