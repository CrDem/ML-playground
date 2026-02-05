// nvcc -arch=sm_75 -ptx matmul.cu -o matmul.ptx

#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

extern "C" __global__ void gemm_kernel_fp16( // block [64, 2, 1]
    const half* __restrict__ A,  // [M x K]
    const half* __restrict__ B,  // [K x N]
    half* C,                     // [M x N]
    int M, int N, int K)
{
    const int blockRow = blockIdx.y * 32;
    const int blockCol = blockIdx.x * 32;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int localTileRow = ty * 16;
    const int localTileCol = (tx / 32) * 16;
    //const int ti = tx / 16;
    //const int tj = tx % 16;

    __shared__ half As[32][32];
    __shared__ half Bs[32][32];

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag;

    wmma::fill_fragment(acc_frag, 0.0f);

    for (int k0 = 0; k0 < K; k0 += 32) {

        // coop load
        int startSmemIndex = tx / 32 + ty * 2;
        for (int localRow = startSmemIndex; localRow < 32; localRow += 4) {
            int localCol = tx % 32;
            int globalRow = blockRow + localRow;
            int globalCol = blockCol + localCol;
            As[localRow][localCol] = (globalRow < M && k0 + localCol < K) ? A[globalRow * K + k0 + localCol] : __float2half(0.0f);
            Bs[localRow][localCol] = (k0 + localRow < K && globalCol < N) ? B[(k0 + localRow) * N + globalCol] : __float2half(0.0f);
        }
        __syncthreads();

        // 1st half K
        wmma::load_matrix_sync(a_frag, &As[localTileRow][0], 32);
        wmma::load_matrix_sync(b_frag, &Bs[0][localTileCol], 32);
        wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);

        // 2nd half K
        wmma::load_matrix_sync(a_frag, &As[localTileRow][16], 32);
        wmma::load_matrix_sync(b_frag, &Bs[16][localTileCol], 32);
        wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        __syncthreads();
    }

    __shared__ float c_tile[32][32];
    wmma::store_matrix_sync(&c_tile[localTileRow][localTileCol], acc_frag, 32, wmma::mem_row_major);
    __syncthreads();

    // coop load
    int startSmemIndex = tx / 32 + ty * 2;
    for (int localRow = startSmemIndex; localRow < 32; localRow += 4) {
        int localCol = tx % 32;
        int globalRow = blockRow + localRow;
        int globalCol = blockCol + localCol;
        if (globalRow < M && globalCol < N) {
            C[globalRow * N + globalCol] = __float2half(c_tile[localRow][localCol]);
        }
    }
}