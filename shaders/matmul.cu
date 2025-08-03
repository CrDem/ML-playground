// nvcc -arch=sm_75 -ptx matmul.cu -o matmul.ptx

#include <cuda_fp16.h>

const int TILE_WIDTH = 32;

extern "C" __global__ void gemm_kernel_fp16(
    const half* __restrict__ A,  // [M x K]
    const half* __restrict__ B,  // [K x N]
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

    __shared__ half sh_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ half sh_B[TILE_WIDTH][TILE_WIDTH];

    float value = 0.0f;
    const int phases = (K + TILE_WIDTH - 1) / TILE_WIDTH;
    for (int phase = 0; phase < phases; phase++)
    {
        // Load Tiles into shared memory
        int A_col = phase * TILE_WIDTH + tx;
        int B_row = phase * TILE_WIDTH + ty;
        sh_A[ty][tx] = (row < M && A_col < K) ? A[row * K + A_col] : __float2half(0.0f);
        sh_B[ty][tx] = (B_row < K && col < N) ? B[B_row * N + col] : __float2half(0.0f);
        __syncthreads();

        // Dot product
        for (int k = 0; k < TILE_WIDTH; k++) {
            value += __half2float(sh_A[ty][k]) * __half2float(sh_B[k][tx]);
        }
        __syncthreads();
    }

    // Assigning calculated value
    if (row < M && col < N)
        C[row * N + col] = __float2half(value);
}