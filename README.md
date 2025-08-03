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
(.venv) D:\work\GEMM>python ./scripts/generate_and_run.py 1024 2048 512
input file created: ./data/matmul_data.safetensors
Computation took 2.3665ms, GFLOPS: 907.2298161842383
[2025-08-03T00:18:20Z INFO  volgaray_gemm] Reading tensors from file
[2025-08-03T00:18:20Z INFO  volgaray_gemm] Parsing tensors to matrixes
[2025-08-03T00:18:20Z INFO  volgaray_gemm] Initialization cuda context
[2025-08-03T00:18:21Z INFO  volgaray_gemm] Binding data
[2025-08-03T00:18:21Z INFO  volgaray_gemm] Loading PTX module
[2025-08-03T00:18:21Z INFO  volgaray_gemm] Launching kernel
[2025-08-03T00:18:21Z INFO  volgaray_gemm] Copying result back
[2025-08-03T00:18:21Z INFO  volgaray_gemm] Saving results
Our result:
tensor([[ -5.5703,  54.5625, -58.3125,  ...,   6.4961,  31.0000,   8.8047],
        [-27.2812,  91.3125,  23.1406,  ...,  77.0000,  -5.8281,   6.7852],
        [ 17.5781, -30.9688,  38.4062,  ...,  -0.9185, -33.4375, -46.0000],
        ...,
        [-18.3594,  47.9688,  20.0156,  ...,  42.5938, -30.6250,   2.1406],
        [  7.3359,  58.1562,  58.7812,  ...,   9.7734, -62.7188,  34.3125],
        [-61.7812,  59.4375,   3.9727,  ...,   2.0000,  -0.8359,  26.0781]],
       dtype=torch.float16)
Torch result:
tensor([[ -5.5703,  54.5625, -58.3125,  ...,   6.4961,  31.0000,   8.8047],
        [-27.2812,  91.3125,  23.1406,  ...,  77.0000,  -5.8281,   6.7852],
        [ 17.5781, -30.9688,  38.4062,  ...,  -0.9185, -33.4375, -46.0000],
        ...,
        [-18.3594,  47.9688,  20.0156,  ...,  42.5938, -30.6250,   2.1406],
        [  7.3359,  58.1562,  58.7812,  ...,   9.7734, -62.7188,  34.3125],
        [-61.7812,  59.4375,   3.9727,  ...,   2.0000,  -0.8359,  26.0781]],
       dtype=torch.float16)
RMSE: 0.000909217138
```