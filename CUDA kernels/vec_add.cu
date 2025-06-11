extern "C" __global__ void vec_add(const float* a, const float* b, float* out, int n) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < n) {
        out[index] = a[index] + b[index];
    }
}