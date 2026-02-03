// nvcc -arch=sm_75 -ptx matmul.cu -o matmul.ptx

#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

extern "C" __global__ void gemm_kernel_fp16(
    const half* __restrict__ A,  // [M x K]
    const half* __restrict__ B,  // [K x N]
    half* C,                     // [M x N]
    int M, int N, int K)
{
    const int tileRow = blockIdx.y * 16;
    const int tileCol = blockIdx.x * 16;
    const int tx = threadIdx.x;
    const int ti = tx / 16;
    const int tj = tx % 16;

    __shared__ half As[16 * 16];
    __shared__ half Bs[16 * 16];

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag;

    wmma::fill_fragment(acc_frag, 0.0f);

    for (int k0 = 0; k0 < K; k0 += 16) {

        // coop load
        for (int i = 0; i < 16; i+=2) {
            int row = tileRow + i + ti;
            int col = tileCol + tj;
            As[(i + ti) * 16 + tj] = (row < M && (k0 + tj) < K) ? A[row * K + k0 + tj] : __float2half(0.0f);
            Bs[(i + ti) * 16 + tj] = (col < N && (k0 + i + ti) < K) ? B[(k0 + i + ti) * N + col] : __float2half(0.0f);
        }

        wmma::load_matrix_sync(a_frag, As, 16);
        wmma::load_matrix_sync(b_frag, Bs, 16);

        wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }

    __shared__ float c_tile[16 * 16];
    wmma::store_matrix_sync(c_tile, acc_frag, 16, wmma::mem_row_major);

    // coop load
    for (int i = 0; i < 16; i+=2) {
        int row = tileRow + i + ti;
        int col = tileCol + tj;
        if (row < M && col < N) {
            C[row * N + col] = __float2half(c_tile[(i + ti) * 16 + tj]);
        }
    }
}