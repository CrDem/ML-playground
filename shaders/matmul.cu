// nvcc -ptx matmul.cu -o matmul.ptx

#include <cuda_fp16.h>

const int TILE_WIDTH = 16;

extern "C" __global__ void gemm_kernel_fp16(
    const half* A,  // [M x K]
    const half* B,  // [K x N]
    half* C,        // [M x N]
    int M, int N, int K) 
{
    //assert(TILE_WIDTH == blockDim.x);
    //assert(TILE_WIDTH == blockDim.y);
    
    const int by = blockIdx.y;
    const int bx = blockIdx.x; 

    const int ty = threadIdx.y;
    const int tx = threadIdx.x; 

    const int row = TILE_WIDTH*by + ty; // row on C
    const int col = TILE_WIDTH*bx + tx; // col on C

    __shared__ float sh_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float sh_B[TILE_WIDTH][TILE_WIDTH];

    const int phases = (K + TILE_WIDTH - 1) / TILE_WIDTH; //A_n_cols = K, C_n_rows = M, C_n_cols = N

    float value = 0;
    for (int phase = 0; phase < phases; phase++)
    {
        // Load Tiles into shared memory
        if ((row < M) && ((phase*TILE_WIDTH+tx) < K))
          sh_A[ty][tx] = A[(row)*K + (phase*TILE_WIDTH+tx)];
        else
          sh_A[ty][tx] = 0.0f;

        if (((phase*TILE_WIDTH + ty) < K) && (col < N))
          sh_B[ty][tx] = B[(phase*TILE_WIDTH + ty)*N + (col)];
        else
          sh_B[ty][tx] = 0.0f;
        __syncthreads();

        // Dot product
        for (int k_phase = 0; k_phase < TILE_WIDTH; k_phase++)
            value += sh_A[ty][k_phase] * sh_B[k_phase][tx];
        __syncthreads();
    }
    // Assigning calculated value
    if ((row < M) && (col < N))
        C[(row)*N + (col)] = value;
}

/* 
TODO:
1. Fused Multiply-Add (FMA)
2. __ldg (Read-Only Data Cache)

#include <cuda_fp16.h>

const int TILE_WIDTH = 16;

extern "C" __global__ void gemm_kernel_fp16(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M, int N, int K,
    half alpha = __float2half(1.0f),  // Параметр alpha
    half beta = __float2half(0.0f)    // Параметр beta
) {
    const int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    const int col = blockIdx.x * TILE_WIDTH + threadIdx.x;

    __shared__ half sh_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ half sh_B[TILE_WIDTH][TILE_WIDTH];

    half value = __float2half(0.0f);
    for (int phase = 0; phase < (K + TILE_WIDTH - 1) / TILE_WIDTH; ++phase) {
        // Загрузка в shared memory
        int A_col = phase * TILE_WIDTH + threadIdx.x;
        int B_row = phase * TILE_WIDTH + threadIdx.y;
        
        sh_A[threadIdx.y][threadIdx.x] = (row < M && A_col < K) ? A[row * K + A_col] : __float2half(0.0f);
        sh_B[threadIdx.y][threadIdx.x] = (B_row < K && col < N) ? B[B_row * N + col] : __float2half(0.0f);
        
        __syncthreads();

        // Вычисление с использованием __half2 (если TILE_WIDTH кратно 2)
        #if __CUDA_ARCH__ >= 530  // Проверка поддержки half2
        if (threadIdx.x % 2 == 0) {
            for (int k = 0; k < TILE_WIDTH; k += 2) {
                __half2 a = *(__half2*)(&sh_A[threadIdx.y][k]);
                __half2 b = *(__half2*)(&sh_B[k][threadIdx.x]);
                value = __hadd(value, __hmul2(a, b).x);
            }
        }
        #else
        for (int k = 0; k < TILE_WIDTH; ++k) {
            value = __hadd(value, __hmul(sh_A[threadIdx.y][k], sh_B[k][threadIdx.x]));
        }
        #endif
        
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = __hfma(alpha, value, __hmul(beta, C[row * N + col]));
    }
}
*/